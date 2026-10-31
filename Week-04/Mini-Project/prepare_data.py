"""Download TrashNet (garythung/trashnet, dataset-resized.zip) and save 5 waste classes."""
import os
import random
import shutil
import zipfile
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download
from PIL import Image

KEEP = ["cardboard", "glass", "metal", "paper", "plastic"]
OUT = "dataset/raw"
TMP = "dataset/_unzipped"
MAX_SIDE = 512
EXT = (".jpg", ".jpeg", ".png")

zip_path = hf_hub_download(repo_id="garythung/trashnet",
                           filename="dataset-resized.zip",
                           repo_type="dataset")
print("Downloaded:", zip_path)

shutil.rmtree(TMP, ignore_errors=True)
shutil.rmtree(OUT, ignore_errors=True)
with zipfile.ZipFile(zip_path) as z:
    z.extractall(TMP)

os.makedirs("images", exist_ok=True)
seen = Counter()
for name in KEEP:
    files = []
    for root, _, fs in os.walk(TMP):
        if "__MACOSX" in root or os.path.basename(root) != name:
            continue
        files += [os.path.join(root, f) for f in fs if f.lower().endswith(EXT)]
    folder = os.path.join(OUT, name)
    os.makedirs(folder, exist_ok=True)
    for path in sorted(files):
        seen[name] += 1
        img = Image.open(path).convert("RGB")
        img.thumbnail((MAX_SIDE, MAX_SIDE))
        img.save(os.path.join(folder, f"{name}_{seen[name]:04d}.jpg"), quality=95)

shutil.rmtree(TMP, ignore_errors=True)

print("\nImages saved per class:")
for n in KEEP:
    print(f" {n:<10}{seen[n]:>5}")
print(f" {'total':<10}{sum(seen.values()):>5}")

rng = random.Random(0)
fig, axes = plt.subplots(5, 5, figsize=(11, 11))
for r, name in enumerate(KEEP):
    files = sorted(os.listdir(os.path.join(OUT, name)))
    pick = rng.sample(files, 5)
    for c, f in enumerate(pick):
        ax = axes[r, c]
        ax.imshow(Image.open(os.path.join(OUT, name, f)))
        ax.axis("off")
        if c == 0:
            ax.set_title(name, loc="left", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig("images/sample_images.png", dpi=110)
print("Saved images/sample_images.png")
