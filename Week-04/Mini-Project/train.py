"""Train three configurations, plot curves, save the best model.

Usage:  python train.py        (20 epochs each)
        python train.py 2      (quick 2-epoch test)
"""
import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from common import (CLASSES, IMG_SIZE, MEAN, STD, WasteDataset, build_model,
                    eval_tf, get_device, train_aug_tf)

EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 20
BATCH = 32
LR = 1e-3
torch.manual_seed(0)
device = get_device()
print("Device:", device, "| epochs per configuration:", EPOCHS)

CONFIGS = {
    "A_baseline":     ("baseline", False),
    "B_improved":     ("improved", False),
    "C_improved_aug": ("improved", True),
}

val_loader = DataLoader(WasteDataset("val", eval_tf), batch_size=64)
loss_fn = nn.CrossEntropyLoss()


def run_epoch(model, loader, opt=None):
    training = opt is not None
    model.train(training)
    total, correct, count = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = loss_fn(out, y)
            if training:
                opt.zero_grad()
                loss.backward()
                opt.step()
            total += loss.item() * len(y)
            correct += (out.argmax(1) == y).sum().item()
            count += len(y)
    return total / count, correct / count


os.makedirs("checkpoints", exist_ok=True)
os.makedirs("images", exist_ok=True)
history, summary = {}, {}

for name, (kind, aug) in CONFIGS.items():
    print(f"\n=== {name} (model: {kind}, augmentation: {aug}) ===")
    train_ds = WasteDataset("train", train_aug_tf if aug else eval_tf)
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
    model = build_model(kind).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    h = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val, best_state, best_epoch = -1.0, None, 0
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        tl, ta = run_epoch(model, train_loader, opt)
        vl, va = run_epoch(model, val_loader)
        for k, v in zip(h, (tl, vl, ta, va)):
            h[k].append(v)
        if va > best_val:
            best_val, best_epoch = va, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"epoch {epoch:>2}/{EPOCHS}  train loss {tl:.3f} acc {ta:.3f} | "
              f"val loss {vl:.3f} acc {va:.3f} | {time.time() - t0:.0f}s")

    torch.save({"model_name": kind, "config": name, "state_dict": best_state,
                "classes": CLASSES, "img_size": IMG_SIZE, "mean": MEAN, "std": STD},
               f"checkpoints/{name}.pth")
    history[name] = h
    summary[name] = {"model": kind, "augmentation": aug, "parameters": n_params,
                     "best_val_acc": best_val, "best_epoch": best_epoch}
    print(f"best validation accuracy {best_val:.3f} at epoch {best_epoch}")

# ---- save the best configuration as waste_cnn.pth -------------------------
best_name = max(summary, key=lambda k: summary[k]["best_val_acc"])
torch.save(torch.load(f"checkpoints/{best_name}.pth"), "waste_cnn.pth")
json.dump({"history": history, "summary": summary, "best_config": best_name,
           "epochs": EPOCHS, "batch_size": BATCH, "learning_rate": LR},
          open("training_results.json", "w"), indent=2)

# ---- curves ----------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True)
for col, (name, h) in enumerate(history.items()):
    ep = range(1, len(h["train_loss"]) + 1)
    axes[0, col].plot(ep, h["train_loss"], label="train")
    axes[0, col].plot(ep, h["val_loss"], label="validation")
    axes[0, col].set_title(name)
    axes[1, col].plot(ep, h["train_acc"], label="train")
    axes[1, col].plot(ep, h["val_acc"], label="validation")
    axes[1, col].set_xlabel("Epoch")
axes[0, 0].set_ylabel("Loss")
axes[1, 0].set_ylabel("Accuracy")
axes[0, 0].legend()
plt.tight_layout()
plt.savefig("images/training_curves.png", dpi=130)

print("\nSUMMARY (validation set)")
for name, s in summary.items():
    print(f"  {name:<16} params {s['parameters']:>8,}  best val acc {s['best_val_acc']:.3f} (epoch {s['best_epoch']})")
print(f"\nBest configuration: {best_name}  ->  saved as waste_cnn.pth")
print("Saved images/training_curves.png and training_results.json")
