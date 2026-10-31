"""Load the saved model and predict on new images.

Usage: python predict.py photo1.jpg photo2.jpg
"""
import sys

import torch
from PIL import Image

from common import CLASSES, build_model, eval_tf

MODEL_PATH = "waste_cnn.pth"
_model = None


def get_model():
    global _model
    if _model is None:
        ckpt = torch.load(MODEL_PATH, map_location="cpu")
        model = build_model(ckpt["model_name"])
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        _model = model
    return _model


@torch.no_grad()
def predict(img):
    """Return {class name: probability} for a PIL image (same preprocessing as training)."""
    x = eval_tf(img.convert("RGB")).unsqueeze(0)
    probs = torch.softmax(get_model()(x), dim=1)[0]
    return {c: float(probs[i]) for i, c in enumerate(CLASSES)}


if __name__ == "__main__":
    for path in sys.argv[1:]:
        scores = predict(Image.open(path))
        top3 = sorted(scores.items(), key=lambda kv: -kv[1])[:3]
        print(path)
        for name, p in top3:
            print(f"   {name:<10}{p * 100:>6.1f}%")
