"""Step 5: evaluate on the test set (run after train.py)."""
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from common import CLASSES, WasteDataset, build_model, eval_tf, get_device

device = get_device()
test_ds = WasteDataset("test", eval_tf)
loader = DataLoader(test_ds, batch_size=64)
y_true = np.array([label for _, label in test_ds.items])
paths = [p for p, _ in test_ds.items]
print(f"Test images: {len(test_ds)}")


def load(path):
    ckpt = torch.load(path, map_location="cpu")
    model = build_model(ckpt["model_name"])
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt


@torch.no_grad()
def predict(model):
    out = []
    for x, _ in loader:
        out.append(torch.softmax(model(x.to(device)), dim=1).cpu())
    return torch.cat(out).numpy()


# ---- 1. all three configurations on the test set ---------------------------
results = {}
print("\nTEST ACCURACY, ALL CONFIGURATIONS")
for name in ("A_baseline", "B_improved", "C_improved_aug"):
    model, _ = load(f"checkpoints/{name}.pth")
    p = predict(model)
    results[name] = float((p.argmax(1) == y_true).mean())
    print(f"  {name:<16} {results[name]:.3f}")

# ---- 2. the final model (chosen by validation accuracy) --------------------
model, ckpt = load("waste_cnn.pth")
final = ckpt["config"]
probs = predict(model)
pred, conf = probs.argmax(1), probs.max(1)
acc = float((pred == y_true).mean())
print(f"\nFINAL MODEL: {final} (chosen by validation accuracy)")
print(f"Test accuracy: {acc:.3f}  ({int((pred == y_true).sum())} of {len(y_true)} correct)")

labels = list(range(len(CLASSES)))
print("\nPER-CLASS RESULTS")
print(classification_report(y_true, pred, labels=labels, target_names=CLASSES, digits=3))
cm = confusion_matrix(y_true, pred, labels=labels)
per_class_acc = cm.diagonal() / cm.sum(axis=1)

# most confused pairs
pairs = []
for i in labels:
    for j in labels:
        if i != j and cm[i, j] > 0:
            pairs.append((int(cm[i, j]), CLASSES[i], CLASSES[j]))
pairs.sort(reverse=True)
print("MOST CONFUSED (true -> predicted)")
for n, t, p in pairs[:4]:
    print(f"  {t} -> {p}: {n} images")

# ---- 3. confusion matrix picture -------------------------------------------
fig, ax = plt.subplots(figsize=(6.5, 5.5))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(labels, CLASSES, rotation=30)
ax.set_yticks(labels, CLASSES)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title(f"Confusion matrix ({final}, test set)")
for i in labels:
    for j in labels:
        ax.text(j, i, cm[i, j], ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black")
fig.colorbar(im)
plt.tight_layout()
plt.savefig("images/confusion_matrix.png", dpi=140)

# ---- 4. do scores guarantee correctness? -----------------------------------
edges = [0.0, 0.5, 0.7, 0.9, 1.0001]
print("\nACCURACY BY CONFIDENCE LEVEL")
bin_rows = []
for lo, hi in zip(edges[:-1], edges[1:]):
    m = (conf >= lo) & (conf < hi)
    a = float((pred[m] == y_true[m]).mean()) if m.any() else float("nan")
    bin_rows.append({"range": f"{lo:.1f}-{min(hi, 1.0):.1f}", "images": int(m.sum()), "accuracy": a})
    print(f"  confidence {lo:.1f}-{min(hi, 1.0):.1f}: {int(m.sum()):>4} images, accuracy {a:.3f}")
over = int(((pred != y_true) & (conf >= 0.9)).sum())
print(f"\nOverconfident mistakes (wrong, with score >= 90%): {over}")

print("\nIF THE APP SAYS 'NOT SURE' BELOW A THRESHOLD")
thr_rows = []
for t in (0.5, 0.6, 0.7, 0.8, 0.9):
    m = conf >= t
    a = float((pred[m] == y_true[m]).mean()) if m.any() else float("nan")
    thr_rows.append({"threshold": t, "answered": float(m.mean()), "accuracy_when_answered": a})
    print(f"  threshold {t:.1f}: answers {m.mean() * 100:>5.1f}% of images, accuracy on those {a:.3f}")

fig, ax = plt.subplots(figsize=(6, 4))
ax.bar([r["range"] for r in bin_rows], [0 if np.isnan(r["accuracy"]) else r["accuracy"] for r in bin_rows])
ax.set_ylim(0, 1)
ax.set_xlabel("Model score (confidence)")
ax.set_ylabel("Accuracy")
ax.set_title("Accuracy by confidence level (test set)")
plt.tight_layout()
plt.savefig("images/confidence_vs_accuracy.png", dpi=140)

# ---- 5. failure examples ----------------------------------------------------
wrong = np.where(pred != y_true)[0]
wrong = wrong[np.argsort(-conf[wrong])][:10]
if len(wrong):
    fig, axes = plt.subplots(2, 5, figsize=(15, 7))
    for ax in axes.flatten():
        ax.axis("off")
    for ax, i in zip(axes.flatten(), wrong):
        ax.imshow(Image.open(paths[i]))
        ax.set_title(f"true: {CLASSES[y_true[i]]}\npredicted: {CLASSES[pred[i]]} ({conf[i] * 100:.0f}%)",
                     fontsize=10, color="red")
    plt.suptitle("Most confident mistakes on the test set")
    plt.tight_layout()
    plt.savefig("images/failure_examples.png", dpi=130)

# ---- 6. save -----------------------------------------------------------------
with open("test_predictions.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["path", "true", "predicted", "confidence"])
    for i in range(len(y_true)):
        w.writerow([paths[i], CLASSES[y_true[i]], CLASSES[pred[i]], f"{conf[i]:.4f}"])
json.dump({"test_images": len(y_true), "test_accuracy_all_configs": results,
           "final_config": final, "final_test_accuracy": acc,
           "per_class_accuracy": {c: float(a) for c, a in zip(CLASSES, per_class_acc)},
           "confusion_matrix": cm.tolist(), "confidence_bins": bin_rows,
           "overconfident_mistakes": over, "threshold_table": thr_rows},
          open("evaluation_results.json", "w"), indent=2)
print("\nSaved images/confusion_matrix.png, images/confidence_vs_accuracy.png, "
      "images/failure_examples.png, test_predictions.csv, evaluation_results.json")
