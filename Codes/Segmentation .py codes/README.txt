U-Net based segmentation pipeline for lung nodule detection in low-dose CT (LDCT) images.

Dataset:
- Source: LIDC-IDRI slices
- Total samples: 15,548 image-mask pairs
- Train split: ~12,438
- Validation split: ~3,110
- Image size: 128 x 128
- Multiple annotations per nodule (up to 4 radiologists)

Ground Truth Construction:
- Consensus threshold used during training: 0.25
  → Includes nodules marked by at least 1 radiologist
  → Designed for high-sensitivity detection (reduce false negatives)

Model:
- Architecture: U-Net
- Encoder: ResNet34 (ImageNet pretrained)
- Attention: SCSE (channel + spatial attention)

Training:
- Epochs: 50
- Batch size: 8
- Loss: BCE + Dice Loss
- Optimizer: Adam (lr = 5e-5)

Evaluation:
- Dice Score (t=0.25): ~0.878
- IoU: ~0.804
- Sensitivity: ~0.904
- Specificity: ~0.999

Multi-Threshold Analysis (Sensitivity Study):
- t = 0.25 → Dice ≈ 0.88 (high sensitivity masks)
- t = 0.5  → Dice ≈ 0.80 (majority consensus)
- t = 0.75 → Dice ≈ 0.72
- t = 1.0  → Dice ≈ 0.66 (strict agreement)

Insights:
- Lower threshold increases detection sensitivity but enlarges mask region
- Higher threshold produces stricter, smaller nodules → lower Dice
- Demonstrates trade-off between sensitivity and precision

Outputs:
- Segmentation masks
- Overlay visualizations
- Best vs worst predictions
- Threshold sensitivity plots