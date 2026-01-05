# Handwritten Digit Recognition (01-70)

![Project Banner](https://img.shields.io/badge/Deep%20Learning-Handwritten%20Digit%20Recognition-blue?style=for-the-badge&logo=tensorflow)

<p align="center">
  <img src="https://img.shields.io/badge/Accuracy-82.86%25-success?style=flat-square" alt="Accuracy"/>
  <img src="https://img.shields.io/badge/Model-MobileNetV2-orange?style=flat-square" alt="Model"/>
  <img src="https://img.shields.io/badge/Classes-70-blue?style=flat-square" alt="Classes"/>
  <img src="https://img.shields.io/badge/Framework-TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white" alt="TensorFlow"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" alt="License"/>
</p>

---

## 📖 Overview

This project implements a robust deep learning system for recognizing **handwritten digits from 01 to 70**. Unlike standard MNIST models that only recognize 0-9, this model handles a much larger and more complex set of 70 distinct classes using a newly collected dataset.

The system leverages **Transfer Learning** with a **MobileNetV2** backbone, enhanced by advanced preprocessing techniques like **CLAHE** (Contrast Limited Adaptive Histogram Equalization) and a novel **multi-patch ensemble voting** strategy during inference to achieve high accuracy on challenging real-world handwritten samples.

---

## ✨ Key Features

- **Advanced Preprocessing**: Uses CLAHE and Gaussian Blur to normalize lighting and reduce noise in handwritten images.
- **Transfer Learning**: Fine-tuned MobileNetV2 (pre-trained on ImageNet) for efficient feature extraction.
- **Ensemble Inference**: Splits test images into multiple overlapping patches (35+ patches) and aggregates predictions to handle variable digit positions and scales.
- **Robust Training**: Implements a two-phase training strategy (Frozen Backbone → Fine-tuning) with strip-based validation splitting.
- **High Performance**: Achieves **82.86% accuracy** on a challenging custom test set.

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| **Deep Learning** | TensorFlow, Keras, MobileNetV2 |
| **Computer Vision** | OpenCV (CLAHE, Image Processing) |
| **Data Handling** | NumPy, Pandas, Scikit-Learn |
| **Language** | Python 3.x |

---

## 🏗️ System Architecture

The pipeline consists of three main stages:

1.  **Preprocessing**:
    *   Convert to Grayscale
    *   Apply CLAHE (Contrast Enhancement)
    *   Resize to 300px height (maintaining aspect ratio)
    *   Convert to 3-channel RGB (for MobileNetV2 compatibility)

2.  **Model (MobileNetV2)**:
    *   **Input**: 300x300x3 images
    *   **Backbone**: MobileNetV2 (weights='imagenet')
    *   **Head**: Global Average Pooling → Dropout (0.5) → Dense (70 classes, Softmax)

3.  **Inference (Ensemble Voting)**:
    *   The input image is sliced into multiple 300x300 patches.
    *   The model predicts probabilities for each patch.
    *   Final prediction is the average of all patch probabilities.

```mermaid
graph TD
    A[Input Image] --> B[Preprocessing (CLAHE + Resize)]
    B --> C{Inference Strategy}
    C -->|Patch 1| D[MobileNetV2]
    C -->|Patch 2| D
    C -->|...| D
    C -->|Patch N| D
    D --> E[Aggregate Probabilities]
    E --> F[Final Prediction (Class 01-70)]
```

---

## 📊 Performance Results

The model was evaluated on a held-out test set of 140 images.

| Metric | Value |
|--------|-------|
| **Total Test Images** | 140 |
| **Correct Predictions** | 116 |
| **Accuracy** | **82.86%** |

> **Note**: The ensemble voting strategy significantly improved accuracy by allowing the model to "see" the digit from multiple positions, making it robust to translation and scaling issues common in handwriting.

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install tensorflow opencv-python pandas scikit-learn numpy
```

### Training the Model

To train the model from scratch using your own dataset:

```bash
python train.py
```
*This script will automatically handle data loading, preprocessing, and the two-phase training process.*

### Running Inference

To test the model on new images in the `test/` folder:

```bash
python run.py
```
*Results will be saved to `result.csv`.*

---

## 📂 Project Structure

```
handwritten-digit-recognition/
├── train/                  # Training images (e.g., 01_1_001.png)
├── test/                   # Test images for evaluation
├── train.py                # Training script (Transfer Learning)
├── run.py                  # Inference script (Ensemble Voting)
├── model.keras             # Saved trained model
├── classes.npy             # Label encoder classes
├── result.csv              # Output predictions
└── README.md               # Project documentation
```

---

## 👨‍💻 Author

**Mansoor Hasan Ali Shokal**
*   [GitHub Profile](https://github.com/mh2des)
*   [LinkedIn](https://www.linkedin.com/in/mansoor-shokla-1a9781353)

---

<p align="center">
  Made with ❤️ and ☕ using TensorFlow
</p>
