import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
import numpy as np

# ================= PATH =================
DATA_DIR = r"D:\Major Project Datasets\IMP Archive\archive\The IQ-OTHNCCD lung cancer dataset\The IQ-OTHNCCD lung cancer dataset"
BATCH_SIZE = 16
EPOCHS = 15
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", DEVICE)

# ================= TRANSFORMS =================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
])

# ================= DATASET =================
dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
print("Classes:", dataset.classes)
print("Total images:", len(dataset))

# ================= SPLIT =================
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_data, val_data = random_split(dataset, [train_size, val_size])
print(f"Train: {train_size} | Val: {val_size}")

# ================= CLASS WEIGHTS =================
targets = [dataset.samples[i][1] for i in train_data.indices]
class_counts = np.bincount(targets)
print("Class counts:", dict(enumerate(class_counts)))
class_weights = 1.0 / class_counts
sample_weights = [class_weights[t] for t in targets]
sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, sampler=sampler)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)

# ================= MODEL =================
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, 3)
model = model.to(DEVICE)

# ================= LOSS =================
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# ================= TRAIN =================
best_acc = 0
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        pred = model(x)
        loss = criterion(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{EPOCHS} Loss: {total_loss:.4f}")
    
    # ================= VALIDATION =================
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = model(x)
            _, predicted = torch.max(pred, 1)
            correct += (predicted == y).sum().item()
            total += y.size(0)
    acc = 100 * correct / total
    print(f"Validation Accuracy: {acc:.2f}%")
    
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "classifier_balanced.pth")
        print(f"  Saved (acc={acc:.2f}%)")

# ================= FINAL =================
print(f"\nDONE - Best Val Accuracy: {best_acc:.2f}%")
print("Model saved as classifier_balanced.pth")