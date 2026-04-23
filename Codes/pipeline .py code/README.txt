End-to-end lung cancer detection pipeline integrating all stages.

Pipeline Flow:
1. Input CT image
2. Denoising (bilateral filter)
3. Segmentation (U-Net → nodule mask)
4. Classification (ResNet18 → benign/malignant/normal)

Outputs:
- Original image
- Denoised image
- Segmentation mask
- Overlay visualization
- Final diagnosis with confidence

Performance:
- Segmentation Dice (t=0.5): ~0.80
- Classification Accuracy: ~99.5%
- End-to-end pipeline accuracy: 9/9 sample cases

Features:
- Fully automated pipeline
- High-confidence predictions (~99%)
- Visual explanation via segmentation mask overlayEnd-to-end lung cancer detection pipeline integrating all stages.

Pipeline Flow:
1. Input CT image
2. Denoising (bilateral filter)
3. Segmentation (U-Net → nodule mask)
4. Classification (ResNet18 → benign/malignant/normal)

Outputs:
- Original image
- Denoised image
- Segmentation mask
- Overlay visualization
- Final diagnosis with confidence

Performance:
- Segmentation Dice (t=0.5): ~0.80
- Classification Accuracy: ~99.5%
- End-to-end pipeline accuracy: 9/9 sample cases

Features:
- Fully automated pipeline
- High-confidence predictions (~99%)
- Visual explanation via segmentation mask overlay