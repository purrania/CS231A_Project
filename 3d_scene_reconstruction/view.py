import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

CLASS_NAMES = ['cube', 'uv_sphere', 'cylinder']

model = fasterrcnn_resnet50_fpn(num_classes=len(CLASS_NAMES)+1)  
model.load_state_dict(torch.load("fasterrcnn_trained.pth"))
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

image_dir = "data/raw/images"

def visualize_prediction(img_path, prediction, threshold=0.5):
    img = Image.open(img_path).convert("RGB")
    plt.figure(figsize=(10,8))
    plt.imshow(img)
    ax = plt.gca()

    boxes = prediction['boxes'].cpu().detach().numpy()
    labels = prediction['labels'].cpu().detach().numpy()
    scores = prediction['scores'].cpu().detach().numpy()

    for box, label, score in zip(boxes, labels, scores):
        if score < threshold:
            continue
        x1, y1, x2, y2 = box
        width, height = x2 - x1, y2 - y1
        rect = patches.Rectangle((x1, y1), width, height, linewidth=2,
                                 edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(x1, y1, f"{CLASS_NAMES[label-1]}: {score:.2f}", 
                fontsize=12, color='white', backgroundcolor='red')

    plt.axis('off')
    plt.show()

def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image_tensor = F.to_tensor(image).to(device)
    with torch.no_grad():
        preds = model([image_tensor])
    return preds[0]

if __name__ == "__main__":
    test_images = os.listdir(image_dir)[:5]  

    for img_name in test_images:
        print(f"Predicting {img_name}...")
        img_path = os.path.join(image_dir, img_name)
        pred = predict_image(img_path)
        visualize_prediction(img_path, pred)

