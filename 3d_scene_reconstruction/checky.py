import json
class_name_to_id = {
    "cube": 0,
    "cylinder": 1,
    "uv_sphere": 2,
}

def convert_bbox_to_yolo(bbox, img_w, img_h):
    x_min, y_min, x_max, y_max = bbox
    x_c = ((x_min + x_max) / 2) / img_w
    y_c = ((y_min + y_max) / 2) / img_h
    w = (x_max - x_min) / img_w
    h = (y_max - y_min) / img_h
    return x_c, y_c, w, h

with open("data/raw/annotations/scene_000.json") as f:
    ann = json.load(f)

img_w, img_h = 640, 480  

labels = []
for bbox, cls_name in zip(ann["bboxes"], ann["classes"]):
    cls_id = class_name_to_id[cls_name]
    x_c, y_c, w, h = convert_bbox_to_yolo(bbox, img_w, img_h)
    labels.append(f"{cls_id} {x_c} {y_c} {w} {h}")

label_file = "data/raw/images/scene_000.txt"
with open(label_file, "w") as f:
    f.write("\n".join(labels))
 
