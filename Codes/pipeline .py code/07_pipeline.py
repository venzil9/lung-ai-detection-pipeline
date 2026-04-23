"""Phase 2 - Script 7: Unified End-to-End Pipeline (FIXED)"""

from pathlib import Path
import numpy as np
from PIL import Image
import cv2
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from torchvision import models, transforms
import matplotlib.pyplot as plt

UNET_PATH       = r"D:\Major Project Datasets\Segmentation Work\Models\best_unet.pth"
CLASSIFIER_PATH = r"D:\Major Project Datasets\Classification Work\Models\classifier_balanced.pth"
OUTPUT_DIR      = Path(r"D:\Major Project Datasets\Classification Work\Results\Pipeline")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES  = ["Benign", "Malignant", "Normal"]
CLASS_COLORS = {"Benign": (255, 165, 0), "Malignant": (255, 0, 0), "Normal": (0, 200, 0)}

print("Loading models...")
unet = smp.Unet(encoder_name="resnet34", encoder_weights=None,
                in_channels=3, classes=1, decoder_attention_type="scse").to(DEVICE)
unet.load_state_dict(torch.load(UNET_PATH, map_location=DEVICE))
unet.eval()

classifier = models.resnet18(weights=None)
classifier.fc = nn.Linear(classifier.fc.in_features, 3)
classifier.load_state_dict(torch.load(CLASSIFIER_PATH, map_location=DEVICE))
classifier = classifier.to(DEVICE)
classifier.eval()
print("Models ready\n")

# EXACT SAME TRANSFORM AS TRAINING (script 06) minus augmentation
cls_tfm = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
])


def run_pipeline(image_path, save_name=None):
    image_path = Path(image_path)
    print(f"=== Processing: {image_path.name} ===")
    
    # Load original using PIL (same as training)
    original_pil = Image.open(image_path).convert("L")
    original_np  = np.array(original_pil)
    H, W = original_np.shape
    
    # STEP 1: Bilateral filter (Objective 1 - denoising)
    print("  [1/3] Bilateral filter...")
    filtered_np = cv2.bilateralFilter(original_np, d=9, sigmaColor=75, sigmaSpace=75)
    
    # STEP 2: U-Net segmentation (Objective 2)
    print("  [2/3] U-Net segmentation...")
    seg_input = cv2.resize(filtered_np, (128, 128))
    seg_norm = seg_input.astype(np.float32) / 255.0
    seg_tensor = torch.tensor(seg_norm).unsqueeze(0).repeat(3,1,1).unsqueeze(0).float().to(DEVICE)
    with torch.no_grad():
        pred = torch.sigmoid(unet(seg_tensor))[0,0].cpu().numpy()
    mask_128 = (pred > 0.5).astype(np.uint8)
    mask_full = cv2.resize(mask_128 * 255, (W, H), interpolation=cv2.INTER_NEAREST)
    mask_full = (mask_full > 127).astype(np.uint8)
    nodule_pixels = int(mask_full.sum())
    has_nodule = nodule_pixels > 20
    print(f"      Nodule pixels: {nodule_pixels}")
    
    # STEP 3: Classification on ORIGINAL image (Objective 3)
    print("  [3/3] Classification...")
    cls_input = cls_tfm(original_pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = classifier(cls_input)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    pred_class = CLASS_NAMES[int(probs.argmax())]
    confidence = float(probs.max())
    print(f"      Result: {pred_class} ({confidence*100:.2f}%)")
    print(f"      Benign={probs[0]:.3f} | Malignant={probs[1]:.3f} | Normal={probs[2]:.3f}")
    
    # Annotated image
    annotated = cv2.cvtColor(original_np, cv2.COLOR_GRAY2RGB)
    if has_nodule and pred_class != "Normal":
        contours, _ = cv2.findContours(mask_full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        color = CLASS_COLORS[pred_class]
        cv2.drawContours(annotated, contours, -1, color, 2)
        if contours:
            biggest = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(biggest)
            cv2.rectangle(annotated, (x, y), (x+w, y+h), color, 2)
    
    # 5-panel visualization
    fig, axes = plt.subplots(1, 5, figsize=(22, 5))
    axes[0].imshow(original_np, cmap='gray')
    axes[0].set_title('1. Original CT', fontweight='bold'); axes[0].axis('off')
    axes[1].imshow(filtered_np, cmap='gray')
    axes[1].set_title('2. Bilateral Filtered', fontweight='bold'); axes[1].axis('off')
    axes[2].imshow(mask_full, cmap='Reds')
    axes[2].set_title('3. U-Net Mask', fontweight='bold'); axes[2].axis('off')
    axes[3].imshow(original_np, cmap='gray'); axes[3].imshow(mask_full, cmap='Reds', alpha=0.4)
    axes[3].set_title('4. Overlay', fontweight='bold'); axes[3].axis('off')
    axes[4].imshow(annotated)
    color_text = {'Benign':'orange','Malignant':'red','Normal':'green'}[pred_class]
    axes[4].set_title(f'5. {pred_class}\n{confidence*100:.1f}% confidence',
                     fontsize=13, fontweight='bold', color=color_text)
    axes[4].axis('off')
    plt.suptitle(f"Lung Cancer Detection Pipeline - {image_path.name}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    out_path = OUTPUT_DIR / (save_name or f"pipeline_{image_path.stem}.png")
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path.name}\n")
    
    return {"name": image_path.name, "prediction": pred_class, "confidence": confidence,
            "nodule": has_nodule, "probs": probs.tolist()}


if __name__ == "__main__":
    DATASET = Path(r"D:\Major Project Datasets\IMP Archive\archive\The IQ-OTHNCCD lung cancer dataset\The IQ-OTHNCCD lung cancer dataset")
    test_images = []
    for cls in ["Benign cases", "Malignant cases", "Normal cases"]:
        files = sorted((DATASET / cls).glob("*"))[:3]
        test_images.extend(files)
    
    print(f"Testing on {len(test_images)} images\n")
    results = [run_pipeline(img) for img in test_images]
    
    print("\n" + "="*85)
    print("PIPELINE RESULTS SUMMARY")
    print("="*85)
    print(f"{'Image':<35}{'Expected':<12}{'Predicted':<12}{'Confidence':<12}{'Match':<8}")
    print("-"*85)
    correct = 0
    for r in results:
        name = r['name']
        name_low = name.lower()
        if 'malig' in name_low:
            expected = "Malignant"
        elif 'beng' in name_low or 'benign' in name_low:
            expected = "Benign"
        else:
            expected = "Normal"
        
        display_name = name[:33]
        match = "YES" if r['prediction'] == expected else "NO"
        if r['prediction'] == expected:
            correct += 1
        print(f"{display_name:<35}{expected:<12}{r['prediction']:<12}{r['confidence']*100:<11.2f}%{match:<8}")
    print("="*85)
    print(f"Pipeline accuracy: {correct}/{len(results)} ({100*correct/len(results):.1f}%)")
    print(f"All outputs in: {OUTPUT_DIR}")