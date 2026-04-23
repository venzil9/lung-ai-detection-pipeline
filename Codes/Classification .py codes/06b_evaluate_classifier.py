"""Phase 2 - Script 6b: Classifier Test Evaluation + Visualizations"""

import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import json

# ============ CONFIG ============
DATA_DIR    = r"D:\Major Project Datasets\IMP Archive\archive\The IQ-OTHNCCD lung cancer dataset\The IQ-OTHNCCD lung cancer dataset"
MODEL_PATH  = r"D:\Major Project Datasets\Classification Work\Models\classifier_balanced.pth"
RESULTS_DIR = Path(r"D:\Major Project Datasets\Classification Work\Results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# Must match training seed for same split
torch.manual_seed(42)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
])

dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
CLASSES = dataset.classes
READABLE = [c.replace(" cases", "") for c in CLASSES]
print(f"Classes: {CLASSES}")
print(f"Total: {len(dataset)}")

# Same 80/20 split as training (using the same seed)
train_size = int(0.8 * len(dataset))
val_size   = len(dataset) - train_size
_, test_data = random_split(dataset, [train_size, val_size])
test_loader = DataLoader(test_data, batch_size=16, shuffle=False)

# Load model
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 3)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()
print("Model loaded")

# Evaluate
all_preds, all_labels, all_probs = [], [], []
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        probs = torch.softmax(out, dim=1)
        all_preds.extend(out.argmax(1).cpu().numpy().tolist())
        all_labels.extend(y.cpu().numpy().tolist())
        all_probs.extend(probs.cpu().numpy().tolist())

all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)
all_probs  = np.array(all_probs)

# Overall accuracy
test_acc = (all_preds == all_labels).mean()
print(f"\nTest Accuracy: {test_acc*100:.2f}%")

# Per-class metrics
print(f"\n{'Class':<15}{'Precision':<12}{'Recall':<10}{'F1':<10}{'Support':<8}")
print("-" * 55)
per_class = {}
for i, name in enumerate(READABLE):
    tp = ((all_preds == i) & (all_labels == i)).sum()
    fp = ((all_preds == i) & (all_labels != i)).sum()
    fn = ((all_preds != i) & (all_labels == i)).sum()
    support = (all_labels == i).sum()
    precision = tp / (tp + fp + 1e-10)
    recall    = tp / (tp + fn + 1e-10)
    f1        = 2 * precision * recall / (precision + recall + 1e-10)
    print(f"{name:<15}{precision:<12.4f}{recall:<10.4f}{f1:<10.4f}{support:<8}")
    per_class[name] = {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "support": int(support)
    }

# Confusion matrix
cm = np.zeros((3, 3), dtype=int)
for t, p in zip(all_labels, all_preds):
    cm[t, p] += 1

print(f"\nConfusion Matrix:")
print(f"{'':<12}{'Pred:'+READABLE[0]:<12}{'Pred:'+READABLE[1]:<14}{'Pred:'+READABLE[2]:<12}")
for i, name in enumerate(READABLE):
    print(f"True:{name:<8}{cm[i,0]:<12}{cm[i,1]:<14}{cm[i,2]:<12}")

# Save metrics
metrics = {
    "test_accuracy": float(test_acc),
    "per_class": per_class,
    "confusion_matrix": cm.tolist(),
    "classes": READABLE
}
with open(RESULTS_DIR / "06b_test_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

with open(RESULTS_DIR / "06b_test_metrics.txt", "w") as f:
    f.write("LUNG CANCER CLASSIFICATION - TEST RESULTS\n")
    f.write("="*60 + "\n\n")
    f.write(f"Model: ResNet18 (ImageNet pretrained, fine-tuned)\n")
    f.write(f"Dataset: IQ-OTHNCCD (Benign / Malignant / Normal)\n")
    f.write(f"Test samples: {len(all_labels)}\n\n")
    f.write(f"Overall Test Accuracy: {test_acc*100:.2f}%\n\n")
    f.write(f"{'Class':<15}{'Precision':<12}{'Recall':<10}{'F1':<10}{'Support':<8}\n")
    for name in READABLE:
        v = per_class[name]
        f.write(f"{name:<15}{v['precision']:<12.4f}{v['recall']:<10.4f}{v['f1']:<10.4f}{v['support']:<8}\n")
    f.write(f"\nConfusion Matrix (rows=true, cols=pred):\n")
    f.write(f"{'':<12}{READABLE[0]:<12}{READABLE[1]:<12}{READABLE[2]:<12}\n")
    for i, name in enumerate(READABLE):
        f.write(f"{name:<12}{cm[i,0]:<12}{cm[i,1]:<12}{cm[i,2]:<12}\n")

# ============ VISUALIZATIONS ============

# 1. Confusion matrix heatmap
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(3)); ax.set_yticks(range(3))
ax.set_xticklabels(READABLE); ax.set_yticklabels(READABLE)
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
ax.set_title(f"Confusion Matrix (Test Acc: {test_acc*100:.2f}%)")
for i in range(3):
    for j in range(3):
        color = "white" if cm[i,j] > cm.max()/2 else "black"
        ax.text(j, i, str(cm[i,j]), ha="center", va="center", color=color, fontsize=16, fontweight="bold")
plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "06b_confusion_matrix.png", dpi=120)
print(f"\nSaved: 06b_confusion_matrix.png")

# 2. Per-class metrics
fig, ax = plt.subplots(figsize=(9, 5))
f1s = [per_class[c]["f1"] for c in READABLE]
precisions = [per_class[c]["precision"] for c in READABLE]
recalls = [per_class[c]["recall"] for c in READABLE]
x = np.arange(3); w = 0.25
ax.bar(x - w, precisions, w, label="Precision", color="#3498db")
ax.bar(x,     recalls,    w, label="Recall",    color="#e74c3c")
ax.bar(x + w, f1s,        w, label="F1-Score",  color="#2ecc71")
ax.set_xticks(x); ax.set_xticklabels(READABLE, fontsize=11)
ax.set_ylim(0, 1.1); ax.legend(fontsize=10); ax.grid(True, alpha=0.3, axis="y")
ax.set_title(f"Per-Class Performance (Overall Acc: {test_acc*100:.2f}%)", fontsize=12)
for i in range(3):
    ax.text(i - w, precisions[i] + 0.02, f"{precisions[i]:.2f}", ha="center", fontsize=9, fontweight="bold")
    ax.text(i,     recalls[i]    + 0.02, f"{recalls[i]:.2f}",    ha="center", fontsize=9, fontweight="bold")
    ax.text(i + w, f1s[i]        + 0.02, f"{f1s[i]:.2f}",        ha="center", fontsize=9, fontweight="bold")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "06b_per_class_metrics.png", dpi=120)
print(f"Saved: 06b_per_class_metrics.png")

print(f"\n{'='*60}")
print(f"CLASSIFIER EVALUATION COMPLETE")
print(f"Test Accuracy: {test_acc*100:.2f}%")
print(f"All files in: {RESULTS_DIR}")
print(f"{'='*60}")