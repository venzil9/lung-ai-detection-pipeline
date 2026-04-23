"""Phase 2 - Script 3: Evaluation + Visualizations"""

import random
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

# ============ CONFIG ============
DATASET_ROOT = Path(r"D:\Major Project Datasets\Segmentation Datasets\LIDC\archive (1)\LIDC-IDRI-slices")
MODELS_DIR   = Path(r"D:\Major Project Datasets\Segmentation Work\Models")
RESULTS_DIR  = Path(r"D:\Major Project Datasets\Segmentation Work\Results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

# Recreate model architecture (same as training)
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=3,
    classes=1,
    decoder_attention_type="scse"
).to(DEVICE)

# Load trained weights
model.load_state_dict(torch.load(MODELS_DIR / "best_unet.pth", map_location=DEVICE))
model.eval()
print("✅ Model loaded")

# Load data (same pairs logic as training)
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
val_pairs = pairs[split:]
test_subset = val_pairs[:500]   # use 500 for evaluation
print(f"Evaluating on {len(test_subset)} held-out samples")

# ============ METRICS ============
total_dice = 0; total_iou = 0; total_sens = 0; total_spec = 0; total_acc = 0
n = 0
samples_for_viz = []

with torch.no_grad():
    for img_path, mask_paths in tqdm(test_subset, desc="Evaluating"):
        img = np.array(Image.open(img_path).convert("L"), dtype=np.float32) / 255.0
        masks = [np.array(Image.open(m).convert("L"), dtype=np.float32) / 255.0 for m in mask_paths]
        gt = (np.mean(masks, axis=0) >= 0.25).astype(np.float32)
        
        img_t = torch.tensor(img).unsqueeze(0).repeat(3,1,1).unsqueeze(0).float().to(DEVICE)
        pred = torch.sigmoid(model(img_t))[0,0].cpu().numpy()
        pred_bin = (pred > 0.5).astype(np.float32)
        
        # Metrics
        inter = (pred_bin * gt).sum()
        union = pred_bin.sum() + gt.sum()
        dice = (2*inter + 1) / (union + 1)
        iou  = (inter + 1) / (pred_bin.sum() + gt.sum() - inter + 1)
        
        tp = inter
        fn = gt.sum() - tp
        fp = pred_bin.sum() - tp
        tn = gt.size - tp - fn - fp
        sens = (tp + 1) / (tp + fn + 1)
        spec = (tn + 1) / (tn + fp + 1)
        acc  = (tp + tn) / gt.size
        
        total_dice += dice; total_iou += iou
        total_sens += sens; total_spec += spec; total_acc += acc
        n += 1
        
        if len(samples_for_viz) < 8:
            samples_for_viz.append((img, gt, pred_bin, dice))

# Averages
avg_dice = total_dice / n
avg_iou  = total_iou / n
avg_sens = total_sens / n
avg_spec = total_spec / n
avg_acc  = total_acc / n

print("\n" + "="*60)
print("TEST SET EVALUATION RESULTS")
print("="*60)
print(f"Dice Score:   {avg_dice:.4f}")
print(f"IoU:          {avg_iou:.4f}")
print(f"Sensitivity:  {avg_sens:.4f}")
print(f"Specificity:  {avg_spec:.4f}")
print(f"Accuracy:     {avg_acc:.4f}")
print("="*60)

# Save metrics json
metrics = {
    "num_samples": n,
    "dice": float(avg_dice),
    "iou": float(avg_iou),
    "sensitivity": float(avg_sens),
    "specificity": float(avg_spec),
    "accuracy": float(avg_acc)
}
with open(RESULTS_DIR / "03_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

# Save metrics text
with open(RESULTS_DIR / "03_metrics.txt", "w") as f:
    f.write("PHASE 2 - OBJECTIVE 2: SEGMENTATION RESULTS\n")
    f.write("="*50 + "\n\n")
    f.write(f"Model: U-Net (ResNet34 encoder, SCSE attention)\n")
    f.write(f"Dataset: LIDC-IDRI (4-radiologist consensus)\n")
    f.write(f"Test samples: {n}\n\n")
    f.write(f"Dice Score:   {avg_dice:.4f}\n")
    f.write(f"IoU:          {avg_iou:.4f}\n")
    f.write(f"Sensitivity:  {avg_sens:.4f}\n")
    f.write(f"Specificity:  {avg_spec:.4f}\n")
    f.write(f"Accuracy:     {avg_acc:.4f}\n")

# ============ VISUALIZATIONS ============
print("\n--- Generating visualizations ---")

# 1. Side-by-side samples (8 examples)
fig, axes = plt.subplots(8, 4, figsize=(16, 28))
for i, (img, gt, pred, dice) in enumerate(samples_for_viz):
    axes[i,0].imshow(img, cmap='gray'); axes[i,0].set_title(f'CT Slice #{i+1}'); axes[i,0].axis('off')
    axes[i,1].imshow(gt, cmap='Reds', alpha=0.9); axes[i,1].set_title('Ground Truth'); axes[i,1].axis('off')
    axes[i,2].imshow(pred, cmap='Blues', alpha=0.9); axes[i,2].set_title(f'Prediction (Dice={dice:.3f})'); axes[i,2].axis('off')
    
    # Overlay
    axes[i,3].imshow(img, cmap='gray')
    axes[i,3].imshow(gt,   cmap='Reds',  alpha=0.4)
    axes[i,3].imshow(pred, cmap='Blues', alpha=0.4)
    axes[i,3].set_title('Overlay: Red=GT, Blue=Pred'); axes[i,3].axis('off')
plt.tight_layout()
plt.savefig(RESULTS_DIR / "03_segmentation_samples.png", dpi=120, bbox_inches='tight')
print("Saved: 03_segmentation_samples.png")

# 2. Metrics bar chart
fig, ax = plt.subplots(figsize=(10, 6))
labels = ['Dice', 'IoU', 'Sensitivity', 'Specificity', 'Accuracy']
values = [avg_dice, avg_iou, avg_sens, avg_spec, avg_acc]
colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6']
bars = ax.bar(labels, values, color=colors)
for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.01, f'{val:.4f}', ha='center', fontsize=12, fontweight='bold')
ax.set_ylim(0, 1.1)
ax.set_ylabel('Score')
ax.set_title('U-Net Segmentation Performance on LIDC-IDRI Test Set', fontsize=13)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(RESULTS_DIR / "03_metrics_chart.png", dpi=120)
print("Saved: 03_metrics_chart.png")

# 3. Best & worst cases
samples_sorted = sorted(samples_for_viz, key=lambda x: x[3], reverse=True)
best = samples_sorted[:3]; worst = samples_sorted[-3:]

fig, axes = plt.subplots(2, 3, figsize=(12, 8))
for i, (img, gt, pred, d) in enumerate(best):
    axes[0,i].imshow(img, cmap='gray')
    axes[0,i].imshow(gt,   cmap='Reds',  alpha=0.4)
    axes[0,i].imshow(pred, cmap='Blues', alpha=0.4)
    axes[0,i].set_title(f'BEST #{i+1}  Dice={d:.3f}', color='green', fontweight='bold'); axes[0,i].axis('off')
for i, (img, gt, pred, d) in enumerate(worst):
    axes[1,i].imshow(img, cmap='gray')
    axes[1,i].imshow(gt,   cmap='Reds',  alpha=0.4)
    axes[1,i].imshow(pred, cmap='Blues', alpha=0.4)
    axes[1,i].set_title(f'WORST #{i+1}  Dice={d:.3f}', color='red', fontweight='bold'); axes[1,i].axis('off')
plt.suptitle('Best vs Worst Predictions (Red=GT, Blue=Pred)', fontsize=14)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "03_best_vs_worst.png", dpi=120)
print("Saved: 03_best_vs_worst.png")

print("\n" + "="*60)
print("✅ EVALUATION COMPLETE")
print(f"Dice: {avg_dice:.4f} | IoU: {avg_iou:.4f}")
print(f"All outputs in: {RESULTS_DIR}")
print("="*60)