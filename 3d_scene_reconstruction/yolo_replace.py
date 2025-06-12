import os
import json

CLASS_MAPPING = {'cube': 0, 'uv_sphere': 1, 'cylinder': 2}
json_dir = "data/raw/annotations"
out_dir = "data/yolo_labels"
os.makedirs(out_dir, exist_ok=True)

for file in os.listdir(json_dir):
    with open(os.path.join(json_dir, file)) as f:
        data = json.load(f)
    
    yolo_lines = []
    for cls, bbox in zip(data["classes"], data["bboxes"]):
        cls_id = CLASS_MAPPING[cls]
        x, y, w, h = bbox  # already normalized
        yolo_lines.append(f"{cls_id} {x} {y} {w} {h}")
    
    out_file = os.path.join(out_dir, file.replace(".json", ".txt"))
    with open(out_file, "w") as f:
        f.write("\n".join(yolo_lines))

