import os
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import json
import numpy as np

CLASS_MAPPING = {'armadillo': 0, 'monkey': 1, 'bunny': 2}
NUM_CLASSES = len(CLASS_MAPPING) + 1  

image_dir = "data/raw/images"
annotation_dir = "data/raw/annotations"

class YoloDataset(Dataset):
    def __init__(self, image_dir, annotation_dir, transforms=None):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.transforms = transforms
        self.image_files = sorted(os.listdir(image_dir))
        self.annotation_files = sorted(os.listdir(annotation_dir))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        ann_path = os.path.join(self.annotation_dir, self.annotation_files[idx])
        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        with open(ann_path) as f:
            annot = json.load(f)

        boxes = []
        labels = []
        for cls_name, bbox in zip(annot["classes"], annot["bboxes"]):
            if cls_name not in CLASS_MAPPING:
                continue
            cls_id = CLASS_MAPPING[cls_name] + 1  
            x_c, y_c, bw, bh = bbox

            xmin = (x_c - bw / 2) * w
            xmax = (x_c + bw / 2) * w
            ymin = (y_c - bh / 2) * h
            ymax = (y_c + bh / 2) * h

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(cls_id)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels}

        if self.transforms:
            img = self.transforms(img)

        return img, target

def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def compute_iou(box1, box2):
    inter_xmin = max(box1[0], box2[0])
    inter_ymin = max(box1[1], box2[1])
    inter_xmax = min(box1[2], box2[2])
    inter_ymax = min(box1[3], box2[3])
    inter_w = max(inter_xmax - inter_xmin, 0)
    inter_h = max(inter_ymax - inter_ymin, 0)
    inter_area = inter_w * inter_h

    area1 = (box1[2]-box1[0])*(box1[3]-box1[1])
    area2 = (box2[2]-box2[0])*(box2[3]-box2[1])
    union_area = area1 + area2 - inter_area

    if union_area == 0:
        return 0
    else:
        return inter_area / union_area

def evaluate_map_coco_style(model, dataloader, device, iou_thresholds=None):
    if iou_thresholds is None:
        iou_thresholds = np.arange(0.5, 1.0, 0.05)  

    model.eval()
    all_precisions = {iou: [] for iou in iou_thresholds}
    with torch.no_grad():
        for imgs, targets in dataloader:
            imgs = list(img.to(device) for img in imgs)
            outputs = model(imgs)

            for target, output in zip(targets, outputs):
                gt_boxes = target['boxes'].cpu().numpy()
                gt_labels = target['labels'].cpu().numpy()
                pred_boxes = output['boxes'].cpu().numpy()
                pred_labels = output['labels'].cpu().numpy()
                pred_scores = output['scores'].cpu().numpy()
                print(f"Predicted boxes count: {len(pred_boxes)}")
                print(f"Predicted scores: {pred_scores}")
                print(f"Ground truth boxes count: {len(gt_boxes)}")

                keep = pred_scores > 0.3
                pred_boxes = pred_boxes[keep]
                pred_labels = pred_labels[keep]

                for iou_thres in iou_thresholds:
                    matched = set()
                    tp = 0
                    for pb, pl in zip(pred_boxes, pred_labels):
                        for i, (gb, gl) in enumerate(zip(gt_boxes, gt_labels)):
                            if i in matched:
                                continue
                            iou = compute_iou(pb, gb)
                            if iou >= iou_thres and pl == gl:
                                tp += 1
                                matched.add(i)
                                break
                    fp = len(pred_boxes) - tp
                    fn = len(gt_boxes) - tp

                    precision = tp / (tp + fp + fn + 1e-6)
                    all_precisions[iou_thres].append(precision)

    model.train()

    avg_precisions = []
    for iou_thres in iou_thresholds:
        if all_precisions[iou_thres]:
            avg_precisions.append(np.mean(all_precisions[iou_thres]))
        else:
            avg_precisions.append(0.0)

    map_05_095 = np.mean(avg_precisions)

    return map_05_095, dict(zip(iou_thresholds, avg_precisions))

if __name__ == "__main__":
    import torchvision.transforms as T
    from torch.optim import Adam
    import matplotlib.pyplot as plt

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")

    transforms = T.Compose([
        T.Resize((416, 416)),
        T.ToTensor()
    ])

    full_dataset = YoloDataset(image_dir, annotation_dir, transforms=transforms)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

    model = get_model(NUM_CLASSES).to(device)
    optimizer = Adam(model.parameters(), lr=1e-4)

    num_epochs = 500
    train_losses = []

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        epoch_loss = 0.0
        num_batches = 0
        num_skipped = 0

        for imgs, targets in train_loader:
            imgs = list(imgs)
            targets = list(targets)

            valid_indices = [i for i, t in enumerate(targets) if t["boxes"].numel() > 0]
            if len(valid_indices) == 0:
                num_skipped += len(targets)
                continue

            imgs = [imgs[i].to(device) for i in valid_indices]
            targets = [{k: v.to(device) for k, v in targets[i].items()} for i in valid_indices]

            loss_dict = model(imgs, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            epoch_loss += losses.item()
            num_batches += 1

        avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0
        train_losses.append(avg_epoch_loss)
        print(f"Epoch {epoch+1} average loss: {avg_epoch_loss:.4f}")
        print(f"Skipped {num_skipped} blank images during training.")

        val_map, val_map_per_iou = evaluate_map_coco_style(model, val_loader, device)
        print(f"Epoch {epoch+1} Validation mAP@0.5:0.95 (approx): {val_map:.4f}")
        for iou_thres, ap in val_map_per_iou.items():
            print(f" - AP@IoU={iou_thres:.2f}: {ap:.4f}")

    torch.save(model.state_dict(), "fasterrcnn_armadillo_monkey_bunny.pth")
    print("\nTraining complete and model saved!")

    plt.plot(range(1, num_epochs + 1), train_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Average Training Loss")
    plt.title("Training Loss Curve")
    plt.show()

