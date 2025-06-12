import os, json
import torch
from torch.utils.data import Dataset
from PIL import Image

CLASS_MAPPING = {'cube': 0, 'uv_sphere': 1, 'cylinder': 2}

class YoloDataset(Dataset):
    def __init__(self, image_dir, annotation_dir, transform=None):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.transform = transform
        self.image_files = sorted(os.listdir(image_dir))
        self.annotation_files = sorted(os.listdir(annotation_dir))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        annotation_path = os.path.join(self.annotation_dir, self.annotation_files[idx])

        image = Image.open(image_path).convert("RGB")
        W, H = image.size  

        with open(annotation_path, 'r') as f:
            annot = json.load(f)

        boxes = []
        labels = []

        for cls_name, bbox in zip(annot["classes"], annot["bboxes"]):
            cls_id = CLASS_MAPPING[cls_name]
            x_center, y_center, width, height = bbox

            xmin = (x_center - width / 2) * W
            ymin = (y_center - height / 2) * H
            xmax = (x_center + width / 2) * W
            ymax = (y_center + height / 2) * H

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(cls_id)

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        if self.transform:
            image = self.transform(image)

        target = {
            'boxes': boxes,
            'labels': labels
        }

        return image, target

