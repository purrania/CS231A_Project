import os
import shutil
from pathlib import Path

raw_image_dir = Path("data/raw/images")
raw_label_dir = Path("data/raw/labels")

train_image_dir = Path("data/images/train")
train_label_dir = Path("data/labels/train")

train_image_dir.mkdir(parents=True, exist_ok=True)
train_label_dir.mkdir(parents=True, exist_ok=True)

for img_file in raw_image_dir.glob("*.png"):
    shutil.copy(img_file, train_image_dir / img_file.name)

for label_file in raw_label_dir.glob("*.txt"):
    shutil.copy(label_file, train_label_dir / label_file.name)

print("done")
