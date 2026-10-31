"""Shared code: preprocessing, dataset and models.
Training, evaluation and the Gradio app all import this file, so the app
always preprocesses images exactly the way training did.
"""
import csv

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

CLASSES = ["cardboard", "glass", "metal", "paper", "plastic"]
IMG_SIZE = 128
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# Used for validation, testing and the app (no randomness)
eval_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

# Used only on training images in configuration C
train_aug_tf = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


class WasteDataset(Dataset):
    """Reads splits.csv and returns (image tensor, class index)."""

    def __init__(self, split, transform, csv_path="splits.csv"):
        self.items = []
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                if row["split"] == split:
                    self.items.append((row["path"], CLASSES.index(row["label"])))
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


class BaselineCNN(nn.Module):
    """Small baseline: 2 plain conv blocks, no BatchNorm, no Dropout."""

    def __init__(self, n_classes=len(CLASSES)):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        side = IMG_SIZE // 4
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * side * side, 64), nn.ReLU(),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def conv_block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1),
        nn.BatchNorm2d(cout),
        nn.ReLU(),
        nn.MaxPool2d(2),
    )


class ImprovedCNN(nn.Module):
    """4 conv blocks with BatchNorm, global average pooling and Dropout."""

    def __init__(self, n_classes=len(CLASSES)):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(3, 32), conv_block(32, 64),
            conv_block(64, 128), conv_block(128, 128),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def build_model(kind):
    return {"baseline": BaselineCNN, "improved": ImprovedCNN}[kind]()


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
