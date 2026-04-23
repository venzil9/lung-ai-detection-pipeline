# AI-Driven CT Lung Nodule Detection Pipeline

An end-to-end deep learning system for lung CT analysis combining denoising, segmentation, and classification into a single, reproducible pipeline.

---

## Overview

The pipeline performs:

1. Denoising using bilateral filtering (selected after multi-metric comparison)
2. Segmentation using U-Net (ResNet34 encoder + attention)
3. Classification using a balanced CNN (ResNet18)

Outputs include:
- Segmentation masks
- Overlay visualizations
- Final prediction (Normal / Benign / Malignant) with confidence

---

## Results

- Segmentation Dice: ~0.80 – 0.87 (threshold dependent)
- Classification Accuracy: ~99.55%
- Pipeline (demo set): 9 / 9 correct

---

## Pipeline

Input CT → Denoising → Segmentation → Classification → Output

---

## Models & Dataset

Due to size limitations, full datasets and trained models are hosted externally:

Download here
https://drive.google.com/drive/folders/1Bl_nS8cAfuC_y7nNxdTA34jr3LdGmEB3?usp=sharing

Includes:
- Segmentation dataset (LIDC-IDRI slices)
- Classification dataset (IQ-OTHNCCD)
- Trained models (.pth)

---

## Sample Outputs

### Pipeline Output
![Pipeline](Results/pipeline/pipeline_1.png)

### Segmentation Output
Segmentation:
Results/segmentation/03_segmentation_samples.png

### Classification Output
Confusion Matrix:
Results/classification/05_confusion_matrix.png

---

## Installation

pip install -r requirements.txt

---

## Usage

python code/pipeline/07_pipeline.py

- Uses images from sample_data/
- You can modify input path inside the script

---

## Project Structure

Codes/
  ├── Preprocessing/
  ├── Segmentation/
  ├── Classification/
  ├── pipeline/

Results/
sample_data/
requirements.txt

---

## Datasets

- Segmentation: LIDC-IDRI (preprocessed 2D slices)
- Classification: IQ-OTHNCCD

Datasets are not included in this repository due to size constraints.

---

## Limitations

- Operates on 2D slices (not full 3D CT volumes)
- Performance depends on dataset distribution
- Not validated for real-world clinical deployment

---

## Future Work

- Patient-level evaluation
- 3D volumetric modeling
- Domain generalization improvements
- Real-time deployment interface

---

## Summary

This repository presents a complete AI pipeline for lung nodule detection, integrating preprocessing, segmentation, and classification with both quantitative evaluation and visual outputs.
