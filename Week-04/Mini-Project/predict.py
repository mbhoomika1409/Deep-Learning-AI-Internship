"""Load the saved models and predict on new images.

Usage: python predict.py photo1.jpg photo2.jpg
"""
import os
import sys

import torch
from PIL import Image

from common import CLASSES, build_model, eval_tf

MODEL_PATH = "waste_cnn.pth"
CONFIGS = [("A_baseline", "A: Baseline CNN"),
           ("B_improved", "B: Improved CNN (final)"),
           ("C_improved_aug", "C: Improved CNN + augmentation")]
_cache = {}


def _load(path):
    if path not in _cache:
        ckpt = torch.load(path, map_location="cpu")
        model = build_model(ckpt["model_name"])
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        _cache[path] = model
    return _cache[path]


def get_model():
    return _load(MODEL_PATH)


@torch.no_grad()
def _scores(model, img):
    x = eval_tf(img.convert("RGB")).unsqueeze(0)   # same preprocessing as training
    probs = torch.softmax(model(x), dim=1)[0]
    return {c: float(probs[i]) for i, c in enumerate(CLASSES)}


def predict(img):
    """{class: score} from the final model (waste_cnn.pth)."""
    return _scores(get_model(), img)


def predict_all(img):
    """{model title: {class: score}} for the baseline, improved and augmented models."""
    out = {}
    for name, title in CONFIGS:
        path = f"checkpoints/{name}.pth"
        if os.path.exists(path):
            out[title] = _scores(_load(path), img)
    return out


if __name__ == "__main__":
    for path in sys.argv[1:]:
        scores = predict(Image.open(path))
        top3 = sorted(scores.items(), key=lambda kv: -kv[1])[:3]
        print(path)
        for name, p in top3:
            print(f"   {name:<10}{p * 100:>6.1f}%")
