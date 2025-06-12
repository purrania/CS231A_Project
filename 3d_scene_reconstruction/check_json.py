import json
import os
from PIL import Image

annotation_dir = "./data/raw/annotations"
image_dir = "./data/raw/images"

def convert_bbox_yolo(img_w, img_h, bbox):
    x_min, y_min, x_max, y_max = bbox
    x_center = (x_min + x_max) / 2.0
    y_center = (y_min + y_max) / 2.0
    width = x_max - x_min
    height = y_max - y_min
    return [x_center / img_w, y_center / img_h, width / img_w, height / img_h]

unique_classes = set()

for ann_file in sorted(os.listdir(annotation_dir)):
    if not ann_file.endswith(".json"):
        continue
    with open(os.path.join(annotation_dir, ann_file)) as f:
        ann = json.load(f)
    unique_classes.update(ann.get("classes", []))

class_map = {cls_name: idx for idx, cls_name in enumerate(sorted(unique_classes))}
print("Discovered classes and their ids:")
for cls_name, cls_id in class_map.items():
    print(f"  {cls_name}: {cls_id}")

for ann_file in sorted(os.listdir(annotation_dir)):
    if not ann_file.endswith(".json"):
        continue
    with open(os.path.join(annotation_dir, ann_file)) as f:
        ann = json.load(f)

    img_path = os.path.join(image_dir, ann["image_path"])
    img = Image.open(img_path)
    w, h = img.size

    yolo_labels = []
    for bbox, cls_name in zip(ann.get("bboxes", []), ann.get("classes", [])):
        cls_id = class_map[cls_name]
        bbox_yolo = convert_bbox_yolo(w, h, bbox)
        yolo_labels.append(f"{cls_id} " + " ".join(map(str, bbox_yolo)))

    label_path = os.path.join(annotation_dir, ann_file.replace(".json", ".txt"))
    with open(label_path, "w") as f:
        f.write("\n".join(yolo_labels))

