# Handwritten Digit Recognition using Transfer Learning

**Author:** Mansoor Hasan Ali Shokal  
**Dataset:** Newly Collected Handwritten Digits  

A deep learning project for recognizing handwritten digits (01-70) using transfer learning with MobileNetV2 architecture. The model achieves **82.86% accuracy** on the test set through advanced preprocessing and ensemble prediction techniques.

## 🎯 Project Overview

This project implements a robust handwritten digit recognition system for **newly collected handwritten digits dataset**. The system uses:
- **Transfer Learning** with MobileNetV2 pre-trained on ImageNet
- **CLAHE (Contrast Limited Adaptive Histogram Equalization)** for preprocessing
- **Multi-patch ensemble prediction** for improved accuracy
- **Data augmentation** for better generalization
- **Train/validation split** with strategic strip-based sampling

## 📊 Model Performance

- **Test Accuracy**: 82.86% (116/140 correct predictions) - **Improved Model**
- **Architecture**: MobileNetV2 (transfer learning)
- **Number of Classes**: 70 (digits 01-70)
- **Input Size**: 300x300 patches
- **Training Strategy**: Two-phase (frozen backbone → fine-tuning)
- **Dataset**: Newly collected handwritten digits

## 🏗️ Architecture

```
Input (300x300x3)
    ↓
Data Augmentation (rotation, translation, zoom, contrast)
    ↓
MobileNetV2 Preprocessing ([-1, 1] normalization)
    ↓
MobileNetV2 Backbone (pre-trained, partially frozen)
    ↓
Global Average Pooling
    ↓
Dropout (0.5)
    ↓
Dense (70 classes, softmax)
```

## 📁 Project Structure

```
DL/
├── train.py              # Training script with transfer learning
├── run.py                # Inference script for testing
├── requirements.txt      # Python dependencies
├── model.keras           # Trained model (saved after training)
├── classes.npy           # Class labels (01-70)
├── result.csv            # Test predictions and results
├── train/                # Training images (70 classes × 1 image each)
│   ├── 01_1_631.png
│   ├── 02_1_632.png
│   └── ...
└── test/                 # Test images (2 images per class)
    ├── 01_2_281.png
    ├── 01_2_491.png
    └── ...
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended for training)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/mh2des/handwritten-digit-recognition.git
cd handwritten-digit-recognition
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Training the Model

To train the model from scratch:

```bash
python train.py
```

**Training Configuration:**
- Batch size: 32
- Head training epochs: 20
- Fine-tuning epochs: 30
- Steps per epoch: 250
- Learning rate: 1e-3 (head), 2e-5 (fine-tuning)
- Label smoothing: 0.05 (head), 0.03 (fine-tuning)

**Training Process:**
1. **Phase 1**: Train classification head with frozen MobileNetV2 backbone
2. **Phase 2**: Fine-tune last 40 layers of backbone (BatchNorm layers remain frozen)

### Testing/Inference

To test the trained model:

```bash
python run.py
```

This will:
- Load the trained model (`model.keras`)
- Process all images in the `test/` directory
- Generate predictions with ensemble voting (35 patches per image)
- Save results to `result.csv`

## 🔍 Key Features

### 1. Advanced Preprocessing
- **CLAHE**: Enhances local contrast for better feature extraction
- **Gaussian Blur**: Reduces noise
- **Aspect-preserving resize**: Maintains original aspect ratio while resizing to 300px height

### 2. Intelligent Patch Extraction
- **Strip-based sampling**: Divides tall images into horizontal strips
- **Train/validation split**: Last 2 strips for validation, rest for training
- **Random patch extraction**: Prevents overfitting to specific image regions

### 3. Multi-Patch Ensemble Prediction
- Extracts 35 patches per test image (15 evenly-spaced + 20 random)
- Processes top and bottom strips separately
- Averages predictions across all patches for robust classification

### 4. Data Augmentation
- Random rotation (±2%)
- Random translation (±5%)
- Random zoom (±5%)
- Random contrast adjustment (±20%)

### 5. Training Optimization
- **Early Stopping**: Monitors validation accuracy with patience=5
- **Learning Rate Reduction**: Reduces LR by 50% when validation plateaus
- **Label Smoothing**: Improves generalization and calibration

## 📈 Results Analysis

The model achieved 82.86% accuracy on 140 test images. Key observations:

- **Strong Performance**: Most classes (01-70) are correctly classified
- **Challenges**: Some confusion between visually similar digits (e.g., 01 vs 70, 35 vs 43)
- **Robustness**: Ensemble prediction significantly improves reliability

### Sample Predictions
See `result.csv` for detailed per-image predictions including:
- `filename`: Test image filename
- `actual`: Ground truth label
- `predicted`: Model prediction

## 🛠️ Technical Details

### Image Format
- **Input**: Grayscale images (variable size)
- **Processing**: Resized to 300px height, preserving aspect ratio
- **Model Input**: 300x300 RGB patches (grayscale converted to 3-channel)

### Model Specifications
- **Base Model**: MobileNetV2 (ImageNet weights)
- **Parameters**: ~2.3M (backbone) + ~89K (classifier head)
- **Training Time**: ~30-50 minutes on GPU (depends on hardware)
- **Inference Time**: ~2-3 seconds per image (with 35-patch ensemble)

## 📦 Dependencies

```
tensorflow>=2.18.0      # Deep learning framework
keras>=3.13.0           # High-level neural networks API
opencv-python>=4.10.0   # Image processing
numpy>=1.26.4           # Numerical computations
pandas>=2.2.3           # Data manipulation and CSV handling
scikit-learn>=1.5.2     # Label encoding
scipy>=1.14.1           # Scientific computing
```

## 🎓 Methodology

### Training Strategy
1. **Transfer Learning**: Leverage MobileNetV2 pre-trained on ImageNet
2. **Two-Phase Training**:
   - Phase 1: Train only the classification head
   - Phase 2: Fine-tune the last layers of the backbone
3. **Validation Strategy**: Hold out last 2 strips per image for validation

### Inference Strategy
1. **Multi-Strip Processing**: Extract top and bottom strips from tall images
2. **Patch Ensemble**: Sample 35 patches from each strip
3. **Probability Averaging**: Combine predictions for final classification

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Implement advanced architectures (EfficientNet, Vision Transformer)
- Add cross-validation for more robust evaluation
- Experiment with different augmentation strategies
- Implement confidence thresholding for uncertain predictions
- Add confusion matrix visualization

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- MobileNetV2 architecture by Google
- ImageNet pre-trained weights
- TensorFlow and Keras teams for excellent frameworks

## 📧 Contact

For questions or suggestions, please open an issue or contact [your-email@example.com]

---

**Note**: The `train/` and `test/` directories should contain your handwritten digit images. Image filenames must follow the format `{label}_{*}.{ext}` where the first two characters represent the class label (01-70).
