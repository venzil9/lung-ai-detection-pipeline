"""Phase 2 - Script 4: Multi-Threshold Sensitivity Analysis"""

import random
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

DATASET_ROOT = Path(r"D:\Major Project Datasets\Segmentation Datasets\LIDC\archive (1)\LIDC-IDRI-slices")
MODELS_DIR   = Path(r"D:\Major Project Datasets\Segmentation Work\Models")
RESULTS_DIR  = Path(r"D:\Major Project Datasets\Segmentation Work\Results")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

# Load model
model = smp.Unet(
    encoder_name="resnet34", encoder_weights=None,
    in_channels=3, classes=1, decoder_attention_type="scse"
).to(DEVICE)
model.load_state_dict(torch.load(MODELS_DIR / "best_unet.pth", map_location=DEVICE))
model.eval()
print("✅ Model loaded")

# Load data (SAME split as training for fair comparison)
pairs = []
for patient in DATASET_ROOT.iterdir():
    if not patient.is_dir(): continue
    for nodule in patient.iterdir():
        if not nodule.is_dir(): continue
        img_dir = nodule / "images"
        if not img_dir.exists(): continue
        for img_path in img_dir.glob("*.png"):
            mask_list = [nodule / f"mask-{i}" / img_path.name for i in range(4)]
            mask_list = [m for m in mask_list if m.exists()]
            if mask_list:
                pairs.append((img_path, mask_list))

random.seed(42)
random.shuffle(pairs)
split = int(0.8 * len(pairs))
val_pairs = pairs[split:][:500]
print(f"Evaluating on {len(val_pairs)} samples")

# ============ MULTI-THRESHOLD EVAL ============
# Only radiologists that actually agree (0.25 = any, 0.5 = majority, 0.75 = strong, 1.0 = all 4)
thresholds = [0.25, 0.50, 0.75, 1.00]
results = {t: {"dice":[], "iou":[], "sens":[], "spec":[], "acc":[]} for t in thresholds}

with torch.no_grad():
    for img_path, mask_paths in tqdm(val_pairs, desc="Multi-threshold eval"):
        img = np.array(Image.open(img_path).convert("L"), dtype=np.float32) / 255.0
        masks = [np.array(Image.open(m).convert("L"), dtype=np.float32) / 255.0 for m in mask_paths]
        mask_mean = np.mean(masks, axis=0)
        
        img_t = torch.tensor(img).unsqueeze(0).repeat(3,1,1).unsqueeze(0).float().to(DEVICE)
        pred = torch.sigmoid(model(img_t))[0,0].cpu().numpy()
        pred_bin = (pred > 0.5).astype(np.float32)
        
        for t in thresholds:
            gt = (mask_mean >= t).astype(np.float32)
            
            # skip if GT is empty at this threshold (makes dice undefined)
            if gt.sum() == 0:
                continue
            
            inter = (pred_bin * gt).sum()
            dice = (2*inter + 1) / (pred_bin.sum() + gt.sum() + 1)
            iou  = (inter + 1) / (pred_bin.sum() + gt.sum() - inter + 1)
            
            tp = inter
            fn = gt.sum() - tp
            fp = pred_bin.sum() - tp
            tn = gt.size - tp - fn - fp
            sens = (tp + 1) / (tp + fn + 1)
            spec = (tn + 1) / (tn + fp + 1)
            acc  = (tp + tn) / gt.size
            
            results[t]["dice"].append(dice)
            results[t]["iou"].append(iou)
            results[t]["sens"].append(sens)
            results[t]["spec"].append(spec)
            results[t]["acc"].append(acc)

# Summarize
summary = {}
print("\n" + "="*70)
print("  MULTI-THRESHOLD SENSITIVITY ANALYSIS")
print("="*70)
print(f"{'Threshold':<12} {'N':<6} {'Dice':<10} {'IoU':<10} {'Sens':<10} {'Spec':<10} {'Acc':<10}")
print("-"*70)
threshold_labels = {
    0.25: "≥1 radiologist (lenient)",
    0.50: "≥2 radiologists (majority)",
    0.75: "≥3 radiologists (strong)",
    1.00: "4/4 radiologists (strict)"
}
for t in thresholds:
    r = results[t]
    n = len(r["dice"])
    d = np.mean(r["dice"]) if n > 0 else 0
    i = np.mean(r["iou"])  if n > 0 else 0
    s = np.mean(r["sens"]) if n > 0 else 0
    p = np.mean(r["spec"]) if n > 0 else 0
    a = np.mean(r["acc"])  if n > 0 else 0
    print(f"t={t:<10} {n:<6} {d:<10.4f} {i:<10.4f} {s:<10.4f} {p:<10.4f} {a:<10.4f}")
    summary[str(t)] = {
        "label": threshold_labels[t],
        "n_samples": n,
        "dice": float(d), "iou": float(i),
        "sensitivity": float(s), "specificity": float(p), "accuracy": float(a)
    }
print("="*70)

# Save
with open(RESULTS_DIR / "04_threshold_analysis.json", "w") as f:
    json.dump(summary, f, indent=2)

with open(RESULTS_DIR / "04_threshold_analysis.txt", "w") as f:
    f.write("MULTI-THRESHOLD SENSITIVITY ANALYSIS\n")
    f.write("="*70 + "\n\n")
    f.write("Same trained model evaluated against ground truths built\n")
    f.write("with varying radiologist-consensus thresholds.\n\n")
    f.write(f"{'Threshold':<12} {'Label':<30} {'Dice':<8} {'IoU':<8} {'Sens':<8} {'Spec':<8}\n")
    f.write("-"*80 + "\n")
    for t in thresholds:
        s = summary[str(t)]
        f.write(f"t={t:<10} {s['label']:<30} {s['dice']:<8.4f} {s['iou']:<8.4f} {s['sensitivity']:<8.4f} {s['specificity']:<8.4f}\n")

# ============ VISUALIZATION ============
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Dice/IoU by threshold
ax = axes[0]
x_labels = [f"t={t}\n({threshold_labels[t].split('(')[0].strip()})" for t in thresholds]
dice_vals = [summary[str(t)]["dice"] for t in thresholds]
iou_vals  = [summary[str(t)]["iou"]  for t in thresholds]
x = np.arange(len(thresholds))
w = 0.35
ax.bar(x - w/2, dice_vals, w, label='Dice', color='#2ecc71')
ax.bar(x + w/2, iou_vals,  w, label='IoU',  color='#3498db')
for i, (d, iou) in enumerate(zip(dice_vals, iou_vals)):
    ax.text(i - w/2, d + 0.01, f'{d:.3f}', ha='center', fontsize=9, fontweight='bold')
    ax.text(i + w/2, iou + 0.01, f'{iou:.3f}', ha='center', fontsize=9, fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(x_labels, fontsize=9)
ax.set_ylabel('Score'); ax.set_ylim(0, 1.05); ax.legend(); ax.grid(True, alpha=0.3, axis='y')
ax.set_title('Dice & IoU vs Consensus Threshold')

# Sens/Spec
ax = axes[1]
sens_vals = [summary[str(t)]["sensitivity"] for t in thresholds]
spec_vals = [summary[str(t)]["specificity"] for t in thresholds]
ax.bar(x - w/2, sens_vals, w, label='Sensitivity', color='#e74c3c')
ax.bar(x + w/2, spec_vals, w, label='Specificity', color='#f39c12')
for i, (s, p) in enumerate(zip(sens_vals, spec_vals)):
    ax.text(i - w/2, s + 0.01, f'{s:.3f}', ha='center', fontsize=9, fontweight='bold')
    ax.text(i + w/2, p + 0.01, f'{p:.3f}', ha='center', fontsize=9, fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(x_labels, fontsize=9)
ax.set_ylabel('Score'); ax.set_ylim(0, 1.1); ax.legend(); ax.grid(True, alpha=0.3, axis='y')
ax.set_title('Sensitivity & Specificity vs Consensus Threshold')

plt.tight_layout()
plt.savefig(RESULTS_DIR / "04_threshold_analysis_chart.png", dpi=120)
print(f"\n✅ Saved: 04_threshold_analysis_chart.png")
print(f"✅ Saved: 04_threshold_analysis.txt")
print(f"✅ Saved: 04_threshold_analysis.json")