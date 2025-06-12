import torch
import torch.nn as nn
from ultralytics import YOLO

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class C2fSE(nn.Module):
    def __init__(self, c2f_block):
        super().__init__()
        self.c2f = c2f_block
        self.se = SEBlock(c2f_block.conv2.out_channels)  
    
    def forward(self, x):
        x = self.c2f(x)
        x = self.se(x)
        return x

def add_se_to_backbone(model):
    backbone = model.model.backbone  

    for i, module in enumerate(backbone.children()):
        if module.__class__.__name__ == "C2f":
            backbone[i] = C2fSE(module)

    model.model.backbone = backbone
    return model

