import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class QuaternionOrientationDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.filenames = sorted(os.listdir(image_dir))
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_name = self.filenames[idx]
        img_path = os.path.join(self.image_dir, img_name)
        label_path = os.path.join(self.label_dir, img_name.replace(".png", ".txt"))

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        with open(label_path, 'r') as f:
            q = list(map(float, f.read().strip().split()))
        quat = torch.tensor(q, dtype=torch.float32)

        return image, quat

class QuaternionOrientationNet(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.resnet18(pretrained=True)
        base.fc = nn.Identity()
        self.backbone = base
        self.head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 4)
        )

    def forward(self, x):
        x = self.backbone(x)
        q = self.head(x)
        return F.normalize(q, dim=1)

def quaternion_loss(pred_q, target_q):
    pred_q = F.normalize(pred_q, dim=1)
    target_q = F.normalize(target_q, dim=1)
    return torch.mean(torch.sum((pred_q - target_q)**2, dim=1))

def train(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    for imgs, quats in dataloader:
        imgs, quats = imgs.to(device), quats.to(device)
        preds = model(imgs)
        loss = quaternion_loss(preds, quats)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


