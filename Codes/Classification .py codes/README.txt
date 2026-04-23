CNN-based classification pipeline for lung CT images.

Dataset:
- Total images: ~1097
- Classes:
  - Benign
  - Malignant
  - Normal
- Class imbalance handled using weighted sampling

Model:
- ResNet18 (pretrained)
- Input: 224x224 (grayscale → 3-channel)

Training:
- Train/Val split: 80/20
- Batch size: 16
- Epochs: 15
- Optimizer: Adam (lr=1e-4)

Techniques:
- WeightedRandomSampler
- Class-weighted CrossEntropy loss
- Data augmentation (flip, rotation)

Performance:
- Test Accuracy: ~99.55%
- Strong performance across all classes after balancing

Outputs:
- Confusion matrix
- Precision / Recall / F1-score