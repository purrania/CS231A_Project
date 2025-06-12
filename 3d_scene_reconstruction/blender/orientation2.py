import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class FourValueOrientationDataset(Dataset):
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
            values = list(map(float, f.read().strip().split()))
        vec4 = torch.tensor(values, dtype=torch.float32)

        return image, vec4

class FourValueOrientationNet(nn.Module):
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
        vec = self.head(x)
        return F.normalize(vec, dim=1)

def four_value_loss(pred, target):
    pred = F.normalize(pred, dim=1)
    target = F.normalize(target, dim=1)
    return torch.mean(torch.sum((pred - target)**2, dim=1))

def train(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    for imgs, vecs in dataloader:
        imgs, vecs = imgs.to(device), vecs.to(device)
        preds = model(imgs)
        loss = four_value_loss(preds, vecs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

