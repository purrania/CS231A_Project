import os
import json
import random

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
print(f"Running splits.py from: {CURRENT_DIR}")

DATA_ROOT = os.path.join(CURRENT_DIR, "data", "raw")  
IMAGE_DIR = os.path.join(DATA_ROOT, "images")
ANNOTATION_DIR = os.path.join(DATA_ROOT, "annotations")

print(f"DATA_ROOT resolved to: {DATA_ROOT}")
print(f"IMAGE_DIR resolved to: {IMAGE_DIR}")

SPLIT_DIR = os.path.join(DATA_ROOT, "splits")
os.makedirs(SPLIT_DIR, exist_ok=True)

def get_scene_ids():
    if not os.path.exists(IMAGE_DIR):
        raise FileNotFoundError(f"Image directory does not exist: {IMAGE_DIR}")
    files = os.listdir(IMAGE_DIR)
    scene_ids = []
    for f in files:
        if f.startswith("scene_") and f.endswith(".png"):
            scene_id = int(f[len("scene_"):-len(".png")])
            scene_ids.append(scene_id)
    scene_ids.sort()
    return scene_ids

def main():
    scene_ids = get_scene_ids()
    print(f"Found {len(scene_ids)} scenes")

    random.seed(42)
    random.shuffle(scene_ids)

    train_ratio = 0.7
    val_ratio = 0.2
    test_ratio = 0.1

    n = len(scene_ids)
    n_train = int(train_ratio * n)
    n_val = int(val_ratio * n)
    n_test = n - n_train - n_val

    train_ids = scene_ids[:n_train]
    val_ids = scene_ids[n_train:n_train + n_val]
    test_ids = scene_ids[n_train + n_val:]

    splits = {
        "train": train_ids,
        "val": val_ids,
        "test": test_ids,
    }

    for split_name, ids in splits.items():
        out_path = os.path.join(SPLIT_DIR, f"{split_name}_split.json")
        with open(out_path, "w") as f:
            json.dump(ids, f, indent=2)
        print(f"Wrote {len(ids)} scene IDs to {out_path}")

if __name__ == "__main__":
    main()

