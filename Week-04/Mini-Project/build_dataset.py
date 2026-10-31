"""Step 2: find duplicate photos, then make a leak-free 70/15/15 split.

Reads  dataset/raw/<class>/*.jpg   (made by prepare_data.py)
Writes splits.csv, dataset_stats.json, images/class_distribution.png,
       images/duplicate_examples.png
"""
import csv
import json
import os
import random

import imagehash
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

RAW = "dataset/raw"
CLASSES = ["cardboard", "glass", "metal", "paper", "plastic"]
FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}
DUP_DIST = 4          # hash distance (out of 64 bits) at or below this = same photo
SEED = 0

# ---- 1. fingerprint every image ------------------------------------------
paths, labels, hashes = [], [], []
for ci, name in enumerate(CLASSES):
    folder = os.path.join(RAW, name)
    for f in sorted(os.listdir(folder)):
        p = os.path.join(folder, f)
        with Image.open(p) as im:
            h = imagehash.phash(im.convert("RGB"))
        paths.append(p)
        labels.append(ci)
        hashes.append(int("".join("1" if b else "0" for b in h.hash.flatten()), 2))
n = len(paths)
labels = np.array(labels)
print(f"Fingerprinted {n} images")

# ---- 2. pairwise Hamming distances ---------------------------------------
h = np.array(hashes, dtype=np.uint64)
POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
dist = np.zeros((n, n), dtype=np.uint8)
for i in range(n):
    x = np.bitwise_xor(h[i], h).view(np.uint8).reshape(n, 8)
    dist[i] = POP[x].sum(axis=1)
np.fill_diagonal(dist, 255)

iu = np.triu_indices(n, k=1)
d_pairs = dist[iu]
print("\nDUPLICATE CHECK")
for t in (0, 2, DUP_DIST, 8):
    print(f"  image pairs with fingerprint distance <= {t}: {int((d_pairs <= t).sum())}")

# ---- 3. group duplicates (union-find) and keep one image per group -------
parent = list(range(n))
def find(a):
    while parent[a] != a:
        parent[a] = parent[parent[a]]
        a = parent[a]
    return a
ii, jj = np.where(np.triu(dist <= DUP_DIST, k=1))
for a, b in zip(ii, jj):
    ra, rb = find(a), find(b)
    if ra != rb:
        parent[rb] = ra
groups = {}
for i in range(n):
    groups.setdefault(find(i), []).append(i)

keep, dropped, mixed = [], 0, 0
for members in groups.values():
    if len(set(labels[members])) > 1:      # same photo under two labels: unusable
        mixed += len(members)
        continue
    keep.append(members[0])
    dropped += len(members) - 1
keep = sorted(keep)
print(f"  duplicate copies removed      : {dropped}")
print(f"  images with conflicting labels: {mixed}")
print(f"  distinct photos kept          : {len(keep)}")

# ---- 4. how bad would a plain random split have been? --------------------
rng = random.Random(SEED)
order = list(range(n))
rng.shuffle(order)
cut = int(n * FRACTIONS["train"])
tr_idx, te_idx = order[:cut], order[-int(n * FRACTIONS["test"]):]
naive_leak = int((dist[np.ix_(te_idx, tr_idx)] <= DUP_DIST).any(axis=1).sum())

# ---- 5. split per class ---------------------------------------------------
split_of = {}
for ci in range(len(CLASSES)):
    idx = [i for i in keep if labels[i] == ci]
    rng.shuffle(idx)
    a = round(len(idx) * FRACTIONS["train"])
    b = a + round(len(idx) * FRACTIONS["val"])
    for i in idx[:a]:
        split_of[i] = "train"
    for i in idx[a:b]:
        split_of[i] = "val"
    for i in idx[b:]:
        split_of[i] = "test"

idx_of = {s: [i for i in keep if split_of[i] == s] for s in FRACTIONS}
def leaks(a, b):
    return int((dist[np.ix_(idx_of[a], idx_of[b])] <= DUP_DIST).any(axis=1).sum())
leak_test, leak_val = leaks("test", "train"), leaks("val", "train")
print("\nLEAK CHECK (images with a twin in train)")
print(f"  plain random split, test set : {naive_leak} of {len(te_idx)}")
print(f"  my split, test set           : {leak_test}")
print(f"  my split, validation set     : {leak_val}")

# ---- 6. class distribution -----------------------------------------------
table = {s: {c: int(sum(labels[i] == k for i in idx_of[s])) for k, c in enumerate(CLASSES)}
         for s in FRACTIONS}
print("\nCLASS DISTRIBUTION")
print(f"{'':<11}" + "".join(f"{s:>7}" for s in FRACTIONS) + f"{'total':>7}")
for c in CLASSES:
    tot = sum(table[s][c] for s in FRACTIONS)
    print(f"{c:<11}" + "".join(f"{table[s][c]:>7}" for s in FRACTIONS) + f"{tot:>7}")
print(f"{'total':<11}" + "".join(f"{len(idx_of[s]):>7}" for s in FRACTIONS) + f"{len(keep):>7}")

# ---- 7. save everything ---------------------------------------------------
with open("splits.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["path", "label", "split"])
    for i in keep:
        w.writerow([paths[i], CLASSES[labels[i]], split_of[i]])

os.makedirs("images", exist_ok=True)
json.dump({"images": n, "distinct_photos": len(keep), "duplicates_removed": dropped,
           "conflicting_label_images": mixed, "dup_distance": DUP_DIST,
           "pairs_at_distance": {str(t): int((d_pairs <= t).sum()) for t in (0, 2, DUP_DIST, 8)},
           "leak_my_split_test": leak_test, "leak_my_split_val": leak_val,
           "leak_random_split_test": naive_leak, "random_test_size": len(te_idx),
           "split_table": table, "classes": CLASSES},
          open("dataset_stats.json", "w"), indent=2)

fig, ax = plt.subplots(figsize=(8, 4.2))
x = np.arange(len(CLASSES))
for k, s in enumerate(FRACTIONS):
    ax.bar(x + (k - 1) * 0.27, [table[s][c] for c in CLASSES], width=0.27, label=s)
ax.set_xticks(x, CLASSES)
ax.set_ylabel("Number of images")
ax.set_title("Class distribution in each split")
ax.legend()
plt.tight_layout()
plt.savefig("images/class_distribution.png", dpi=140)

close = [(a, b) for a, b in zip(ii, jj)][:4]
if close:
    k = len(close)
    fig, axes = plt.subplots(2, k, figsize=(3 * k, 5.4), squeeze=False)
    for col, pair in enumerate(close):
        for row, p in enumerate(pair):
            axes[row, col].imshow(Image.open(paths[p]))
            axes[row, col].set_title(os.path.basename(paths[p]), fontsize=8)
            axes[row, col].axis("off")
    plt.suptitle("Duplicate photos found by the fingerprint check")
    plt.tight_layout()
    plt.savefig("images/duplicate_examples.png", dpi=130)
print("\nSaved splits.csv, dataset_stats.json, images/class_distribution.png"
      + (", images/duplicate_examples.png" if close else ""))
