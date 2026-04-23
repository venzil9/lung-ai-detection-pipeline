# AI-Driven CT Lung Nodule Detection Pipeline

An end-to-end deep learning system for lung CT analysis that combines **denoising**, **segmentation**, and **classification** into a single, reproducible pipeline.

---

## Overview

The pipeline performs:

1. **Denoising** using bilateral filtering
2. **Segmentation** using U-Net (ResNet34 encoder with attention)
3. **Classification** using a balanced CNN (ResNet18)

Outputs include **segmentation masks**, **overlays**, and a **final prediction with confidence**.

---

## Results

* Segmentation Dice: ~0.80–0.87 (threshold dependent)
* Classification Accuracy: ~99.55%
* Pipeline (demo set): 9/9 correct

---

## Pipeline

```
Input CT → Denoising → Segmentation → Classification → Output
```

---

## Sample Outputs

### Pipeline

![Pipeline](results/pipeline/pipeline_output_1.png)

### Segmentation

![Segmentation](results/segmentation/03_segmentation_samples.png)

### Classification

![Confusion Matrix](results/classification/06b_confusion_matrix.png)

---

## Installation

```
pip install -r requirements.txt
```

---

## Usage

```
python code/pipeline/07_pipeline.py
```

Use images from `sample_data/` or provide your own input path in the script.

---

## Project Structure

```
code/
  ├── preprocessing/
  ├── segmentation/
  ├── classification/
  ├── pipeline/

models/
results/
sample_data/
requirements.txt
```

---

## Datasets

* Segmentation: LIDC-IDRI (preprocessed slices)
* Classification: IQ-OTHNCCD

Datasets are not included due to size. Sample inputs are provided.

---

## Limitations

* Operates on 2D slices (not full 3D CT volumes)
* Performance is dataset-dependent
* Not intended for clinical use

---

## Future Work

* Patient-level validation
* 3D volumetric modeling
* Improved generalization
* Deployment-ready interface

---

## Summary

This repository provides a complete, modular pipeline for lung CT analysis with both quantitative metrics and visual interpretability.
