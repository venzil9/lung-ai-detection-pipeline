import random
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
from tqdm import tqdm

DATASET_ROOT = Path(r"D:\Major Project Datasets\Segmentation Datasets\LIDC\archive (1)\LIDC-IDRI-slices")
MODELS_DIR = Path(r"D:\Major Project Datasets\Segmentation Work\Models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", DEVICE)

BATCH_SIZE = 8
EPOCHS = 50
LR = 5e-5

pairs = []

for patient in DATASET_ROOT.iterdir():
    if not patient.is_dir():
        continue

    for nodule in patient.iterdir():
        if not nodule.is_dir():
            continue

        img_dir = nodule / "images"
        if not img_dir.exists():
            continue

        for img_path in img_dir.glob("*.png"):
            mask_list = []
            for i in range(4):
                m = nodule / f"mask-{i}" / img_path.name
                if m.exists():
                    mask_list.append(m)

            if len(mask_list) > 0:
                pairs.append((img_path, mask_list))

random.shuffle(pairs)

split = int(0.8 * len(pairs))
train_pairs = pairs[:split]
val_pairs = pairs[split:]

print("Train:", len(train_pairs), "Val:", len(val_pairs))

class LIDC(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        img_path, masks = self.data[i]

        img = np.array(Image.open(img_path).convert("L"), dtype=np.float32) / 255.0

        mask_arr = []
        for m in masks:
            mask_arr.append(np.array(Image.open(m).convert("L"), dtype=np.float32) / 255.0)

        gt = (np.mean(mask_arr, axis=0) >= 0.25).astype(np.float32)

        img = torch.tensor(img).unsqueeze(0).repeat(3,1,1)
        gt = torch.tensor(gt).unsqueeze(0)

        return img.float(), gt.float()

train_loader = DataLoader(LIDC(train_pairs), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(LIDC(val_pairs), batch_size=BATCH_SIZE)

model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1,
    decoder_attention_type="scse"
).to(DEVICE)

bce = nn.BCEWithLogitsLoss()

def dice_loss(pred, target):
    pred = torch.sigmoid(pred)
    inter = (pred * target).sum()
    return 1 - (2*inter + 1) / (pred.sum() + target.sum() + 1)

def loss_fn(pred, target):
    return 0.5 * bce(pred, target) + 0.5 * dice_loss(pred, target)

opt = torch.optim.Adam(model.parameters(), lr=LR)

best = 0

for epoch in range(EPOCHS):
    model.train()

    for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        x, y = x.to(DEVICE), y.to(DEVICE)

        pred = model(x)
        loss = loss_fn(pred, y)

        opt.zero_grad()
        loss.backward()
        opt.step()

    model.eval()
    total = 0

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            pred = torch.sigmoid(model(x))
            pred = (pred > 0.5).float()

            inter = (pred * y).sum()
            dice = (2*inter + 1) / (pred.sum() + y.sum() + 1)

            total += dice.item()

    val_dice = total / len(val_loader)
    print("Epoch", epoch+1, "Dice:", val_dice)

    if val_dice > best:
        best = val_dice
        torch.save(model.state_dict(), MODELS_DIR / "best_unet.pth")
        print("Saved best")

print("DONE. Best Dice:", best)
