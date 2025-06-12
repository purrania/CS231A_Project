import os
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torchvision.transforms as T

CLASS_NAMES = ['background', 'cube', 'uv_sphere', 'cylinder']

def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def visualize_prediction(image_path, prediction, threshold=0.5):
    img = Image.open(image_path).convert("RGB")
    plt.figure(figsize=(8,6))
    plt.imshow(img)
    ax = plt.gca()

    boxes = prediction['boxes'].cpu().detach().numpy()
    labels = prediction['labels'].cpu().detach().numpy()
    scores = prediction['scores'].cpu().detach().numpy()

    for box, label, score in zip(boxes, labels, scores):
        if score < threshold:
            continue
        xmin, ymin, xmax, ymax = box
        width, height = xmax - xmin, ymax - ymin
        rect = patches.Rectangle((xmin, ymin), width, height, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(xmin, ymin-10, f"{CLASS_NAMES[label]}: {score:.2f}", color='yellow', fontsize=12, bbox=dict(facecolor='red', alpha=0.5))

    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    import sys

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")

    model = get_model(num_classes=len(CLASS_NAMES))
    model.load_state_dict(torch.load("fasterrcnn_trained.pth", map_location=device))
    model.to(device)
    model.eval()
    print("Model loaded.")

    transform = T.Compose([
        T.Resize((416,416)),
        T.ToTensor()
    ])

    image_dir = "data/raw/images"
    test_images = sorted(os.listdir(image_dir))[:5]  

    for img_name in test_images:
        img_path = os.path.join(image_dir, img_name)
        img = Image.open(img_path).convert("RGB")
        input_img = transform(img).to(device)
        with torch.no_grad():
            pred = model([input_img])[0]
        print(f"Predictions for {img_name}:")
        print("Boxes:", pred['boxes'])
        print("Labels:", pred['labels'])
        print("Scores:", pred['scores'])
        visualize_prediction(img_path, pred, threshold=0.3)

