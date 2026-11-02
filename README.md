# Benthic Habitat Segmentation

A lightweight U-Net in PyTorch for semantic segmentation of benthic (seafloor) habitats from 8-band WorldView-3 satellite imagery of Guam. Built for CISC 484/684: Machine Learning at the University of Delaware.

## Overview

The model classifies every pixel into one of seven classes (Background, Seagrass, Coral, Macroalgae, Sand, Land, Ocean) across two study regions:

- **Agana Bay:** shallow coastal waters (1-5 m) with mixed seagrass beds and patchy coral. Key factors: urban runoff, moderate turbidity.
- **Manell-Geus:** barrier reef system (5-25 m) with depth-dependent habitat zonation. Key factors: wave energy, substrate type.

## Dataset

Provided by the course and **not included in this repository**.

| Item | Details |
|---|---|
| Imagery | WorldView-3, 8 bands (Coastal, Blue, Green, Yellow, Red, Red Edge, NIR1, NIR2) |
| Labeled images | 9 GeoTIFFs (6 Agana Bay, 3 Manell-Geus) |
| Test images | 21 (15 Agana Bay, 6 Manell-Geus) |
| Image sizes | 528x447 to 1370x2051 px |

Class distribution (by pixels) is heavily imbalanced:

| Class | Share |
|---|---|
| Land | 37.89% |
| Ocean | 27.41% |
| Macroalgae | 11.91% |
| Sand | 8.60% |
| Coral | 7.65% |
| Background | 3.50% |
| Seagrass | 3.06% |

## Approach

**Preprocessing**
- Variable-sized images are tiled into 64x64 patches with a 48-pixel stride; edge patches smaller than 64x64 are zero-padded.
- Per-band normalization: 2nd-98th percentile clipping, then [0, 1] scaling. This is robust to outliers such as sun glint and cloud shadows.

**Model**
- Lightweight U-Net (119,415 parameters) with an 8-channel input, two encoder stages, a bottleneck, and two decoder stages with skip connections. Skip connections preserve spatial detail for thin habitat boundaries like reef edges.
- Channels: 8 -> 16 -> 32 -> 64 (bottleneck) -> 32 -> 16 -> 7 classes. Each block uses DoubleConv + BatchNorm + ReLU.

**Training**
- Loss: cross-entropy with inverse-frequency class weights to counter class imbalance (e.g., Seagrass 0.32 vs. Land 0.03).

| Parameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Scheduler | CosineAnnealingLR |
| Epochs | 20 |
| Batch size | 2 |
| Patch size | 64x64 |
| Training patches | 300 (subsampled for CPU training) |

Inference uses full-image sliding-window prediction.

## Results

**Labeled images** (these images were also used for training, so this is not held-out performance): mean pixel accuracy **62.0%**, mean IoU **34.5%**.

| Image | Region | Pixel Acc. | mIoU |
|---|---|---|---|
| WV02052023 | Agana Bay | 46.6% | 28.8% |
| WV02202022 | Manell-Geus | 74.0% | 32.1% |
| WV220101215 | Agana Bay | 42.1% | 27.4% |
| WV220130115 | Agana Bay | 57.3% | 28.6% |
| WV220140205 | Agana Bay | 81.2% | 48.6% |
| WV220171020 | Agana Bay | 40.8% | 23.6% |
| WV220230219 | Agana Bay | 77.2% | 48.1% |
| WV320141230 | Manell-Geus | 72.8% | 38.6% |
| WV_01132022 | Manell-Geus | 65.8% | 34.5% |

**Unseen test images:**

| Region | Images | Mean Acc. | Mean mIoU |
|---|---|---|---|
| Agana Bay | 15 | 17.4% | 3.3% |
| Manell-Geus | 6 | 26.1% | 9.9% |

**What worked:** Land (90% recall) and Macroalgae (85% recall) are segmented well across both regions.

**Limitations:**
- Seagrass is almost entirely misclassified as Macroalgae (88%) because of spectral similarity and extreme class imbalance.
- Coral is split between Background (29%), Macroalgae (29%), and correct (24%).
- Ocean is often predicted as Background (59%), and patch-boundary effects create a grid artifact in deep-water regions.
- Performance drops sharply on test images, attributed to temporal domain shift (images span 2010-2024, with differing water clarity, turbidity, and sun angle).

Training was limited by CPU-only compute, which is why the model, patch size, and dataset subset are small.

<!-- Add figures here once they are in the repo, e.g.:
![Training curves](results/training_curves.png)
![Prediction vs. ground truth](results/prediction_example.png)
![Confusion matrix](results/confusion_matrix.png)
-->

## Future Work

- Train longer on more patches with a larger model on a GPU
- Add Dice loss to improve recall on minority classes
- Test-time augmentation (flips/rotations) to reduce patch-boundary artifacts
- Spectral-jitter augmentation to generalize across acquisition dates

## Project Structure

<!-- Adjust to match your actual folders and filenames. -->

```
.
├── Dataset/             # course dataset (not tracked)
├── notebook/
│   ├── seg_final.ipynb  # segmentation notebook
│   └── CNN-1.ipynb
├── Report/
│   ├── report.pdf       # written report
│   ├── v4.py            # training / inference script
│   └── best_unet8.pth   # trained U-Net weights
├── output_masks/        # generated masks (not tracked)
├── predicted_masks/     # test-set predictions (not tracked)
├── .gitignore
└── README.md
```

## Setup

```bash
git clone https://github.com/MelloFlare/REPO-NAME.git
cd REPO-NAME
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Main dependencies: PyTorch, rasterio, NumPy, matplotlib.

<!-- Add the exact commands for training and prediction here. -->

## Tech Stack

Python, PyTorch, U-Net, Rasterio, GeoTIFF, NumPy

## Author

Dhir Patel ([@MelloFlare](https://github.com/MelloFlare))

Course: CISC 484/684 Machine Learning, University of Delaware (Instructor: Dr. Xu Yuan). Dataset provided by the course.
