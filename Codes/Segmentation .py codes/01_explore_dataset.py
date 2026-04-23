"""Phase 2 - Script 1: Dataset Exploration"""

import os
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

DATASET_ROOT = Path(r"D:\Major Project Datasets\Segmentation Datasets\\LIDC\\archive (1)\\LIDC-IDRI-slices")
OUTPUT_DIR   = Path(r"D:\Major Project Datasets\Segmentation Work\Results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*60)
print("LIDC-IDRI DATASET EXPLORATION")
print("="*60)
print(f"Dataset root: {DATASET_ROOT}")
print(f"Exists: {DATASET_ROOT.exists()}\n")

if not DATASET_ROOT.exists():
    raise FileNotFoundError(f"Dataset not found at {DATASET_ROOT}")

patient_folders = sorted([p for p in DATASET_ROOT.iterdir() if p.is_dir() and p.name.startswith("LIDC-IDRI")])
print(f"Total patient folders: {len(patient_folders)}")

total_nodules = 0
total_slices = 0
image_sizes = []
available_pairs = []

for patient in patient_folders:
    nodule_folders = sorted([n for n in patient.iterdir() if n.is_dir()])
    total_nodules += len(nodule_folders)
    for nodule in nodule_folders:
        images_dir = nodule / "images"
        if not images_dir.exists():
            continue
        image_files = sorted(images_dir.glob("*.png"))
        for img_path in image_files:
            mask_paths = []
            for mi in range(4):
                mp = nodule / f"mask-{mi}" / img_path.name
                if mp.exists():
                    mask_paths.append(mp)
            if len(mask_paths) > 0:
                available_pairs.append((img_path, mask_paths))
                total_slices += 1

if available_pairs:
    sample_img = Image.open(available_pairs[0][0])
    image_sizes.append(sample_img.size)

print(f"\nTotal nodules: {total_nodules}")
print(f"Total image-mask pairs: {total_slices}")
print(f"Image size: {image_sizes[0] if image_sizes else 'N/A'}")

for patient in patient_folders[:5]:
    nods = sorted([n for n in patient.iterdir() if n.is_dir()])
    total = 0
    for nod in nods:
        if (nod / "images").exists():
            total += len(list((nod / "images").glob("*.png")))
    print(f"  {patient.name}: {len(nods)} nodule(s), {total} slice(s)")

print("\n--- Saving sample visualization ---")
sample_img_path, sample_mask_paths = available_pairs[0]
img = np.array(Image.open(sample_img_path).convert('L'))
masks = [np.array(Image.open(mp).convert('L')) > 127 for mp in sample_mask_paths]
consensus = (np.mean(masks, axis=0) >= 0.5).astype(np.uint8) * 255

fig, axes = plt.subplots(1, 6, figsize=(18, 3))
axes[0].imshow(img, cmap='gray'); axes[0].set_title('CT Slice'); axes[0].axis('off')
for i, m in enumerate(masks):
    axes[i+1].imshow(m, cmap='gray'); axes[i+1].set_title(f'Radiologist {i+1}'); axes[i+1].axis('off')
axes[5].imshow(consensus, cmap='gray'); axes[5].set_title('CONSENSUS'); axes[5].axis('off')
plt.tight_layout()
out_path = OUTPUT_DIR / "01_sample_with_consensus.png"
plt.savefig(out_path, dpi=120, bbox_inches='tight')
print(f"Saved: {out_path}")

summary_path = OUTPUT_DIR / "01_dataset_summary.txt"
with open(summary_path, 'w') as f:
    f.write(f"LIDC-IDRI Dataset Summary\n{'='*40}\n")
    f.write(f"Total patients: {len(patient_folders)}\n")
    f.write(f"Total nodules: {total_nodules}\n")
    f.write(f"Total image-mask pairs: {total_slices}\n")
    f.write(f"Image size: {image_sizes[0] if image_sizes else 'N/A'}\n")

print(f"\n✅ DONE. {total_slices} training pairs ready from {len(patient_folders)} patients")
