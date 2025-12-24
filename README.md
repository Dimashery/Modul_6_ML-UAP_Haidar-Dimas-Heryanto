# 🎭 Emotion Recognition System
### Deep Learning-Based Facial Emotion Classification

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://modul6ml-uaphaidar-dimas-heryanto.streamlit.app/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.15+](https://img.shields.io/badge/TensorFlow-2.15+-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<div align="center">
  <img src="https://img.shields.io/badge/Accuracy-87%25-brightgreen" alt="Best Accuracy">
  <img src="https://img.shields.io/badge/Models-4-blue" alt="Models">
  <img src="https://img.shields.io/badge/Classes-3-purple" alt="Classes">
</div>

---

## 📋 Table of Contents
- [Overview](#-overview)
- [Dataset](#-dataset)
- [Models Architecture](#-models-architecture)
- [Performance Comparison](#-performance-comparison)
- [Key Features](#-key-features)
- [Installation](#-installation)
- [Usage](#-usage)
- [Results & Analysis](#-results--analysis)
- [Live Demo](#-live-demo)
- [Project Structure](#-project-structure)
- [Technologies Used](#-technologies-used)
- [Contributors](#-contributors)

---

## 🎯 Overview

This project implements a **comprehensive emotion recognition system** using deep learning techniques to classify facial expressions into three primary emotions: **Angry**, **Sad**, and **Surprise**. The system compares four different neural network architectures to identify the most effective model for emotion detection.

### 🎓 Academic Context
- **Course**: Machine Learning - Semester 7
- **Institution**: [Your University Name]
- **Project Type**: Final Assignment (UAP)

### 🌟 Project Highlights
- ✅ Four state-of-the-art deep learning models
- ✅ Comprehensive error analysis and visualization
- ✅ Interactive web-based interface
- ✅ Batch and single image prediction
- ✅ Real-time emotion detection
- ✅ Detailed performance metrics

---

## 📊 Dataset

### Dataset Information
- **Source**: [Kaggle - Emotion Recognition Dataset](https://www.kaggle.com/datasets/sujaykapadnis/emotion-recognition-dataset)
- **Total Images**: 6,470 images
- **Classes**: 3 emotions (Angry, Sad, Surprise)
- **Image Format**: Grayscale/RGB facial images
- **Image Size**: Variable (resized to 128x128 for training)

### Class Distribution

| Emotion | Training Samples | Validation Samples | Total |
|---------|-----------------|-------------------|--------|
| **Angry** | 210 | 52 | 262 |
| **Sad** | 629 | 157 | 786 |
| **Surprise** | 197 | 49 | 246 |
| **Total** | 1,036 | 258 | **1,294** |

### 📈 Data Distribution Visualization
```
Sad (60.7%)     ████████████████████████████████████████████████
Angry (20.2%)   ████████████████
Surprise (19.1%) ███████████████
```

### ⚖️ Class Imbalance Handling
To address the class imbalance, we implemented:
- **Class Weights**: Computed weights inversely proportional to class frequencies
  - Angry: 1.65
  - Sad: 0.55
  - Surprise: 1.75
- **Data Augmentation**: Enhanced minority class representation

---

## 🏗️ Models Architecture

### 1️⃣ Custom CNN (Convolutional Neural Network)

**Architecture Overview:**
```
Input (128x128x3)
    ↓
Conv2D(32) → BatchNorm → MaxPool → Dropout(0.25)
    ↓
Conv2D(64) → BatchNorm → MaxPool → Dropout(0.25)
    ↓
Conv2D(128) → BatchNorm → MaxPool → Dropout(0.25)
    ↓
Conv2D(256) → BatchNorm → GlobalAvgPool
    ↓
Dense(256) → BatchNorm → Dropout(0.5)
    ↓
Dense(128) → Dropout(0.3)
    ↓
Dense(3, softmax)
```

**Key Features:**
- 4 Convolutional blocks with increasing filters (32 → 64 → 128 → 256)
- Batch Normalization after each conv layer
- L2 Regularization (0.001)
- Global Average Pooling to reduce overfitting
- Total Parameters: ~3.5M

**Training Configuration:**
- Optimizer: Adam (lr=0.001)
- Loss: Categorical Crossentropy
- Epochs: 30 (with EarlyStopping)
- Batch Size: 32

---

### 2️⃣ ResNet50 (Residual Network)

**Architecture Overview:**
```
ResNet50 (ImageNet Pretrained)
    ↓
GlobalAveragePooling2D
    ↓
BatchNormalization
    ↓
Dense(512, relu, L2) → Dropout(0.5) → BatchNorm
    ↓
Dense(256, relu, L2) → Dropout(0.4) → BatchNorm
    ↓
Dense(128, relu, L2) → Dropout(0.3)
    ↓
Dense(3, softmax)
```

**Transfer Learning Strategy:**
- **Stage 1**: Train custom head (frozen base) - 15 epochs, LR: 1e-4
- **Stage 2**: Fine-tune top 30 layers - 25 epochs, LR: 1e-5
- Total Parameters: ~25.6M
- Trainable Parameters: ~8.2M

**Preprocessing:**
- Uses ResNet50-specific preprocessing (`preprocess_input`)
- Mean subtraction and scaling based on ImageNet

---

### 3️⃣ MobileNetV2 (Efficient Mobile Architecture)

**Architecture Overview:**
```
MobileNetV2 (ImageNet Pretrained)
    ↓
GlobalAveragePooling2D
    ↓
BatchNormalization
    ↓
Dense(512, relu, L2) → Dropout(0.5) → BatchNorm
    ↓
Dense(256, relu, L2) → Dropout(0.4) → BatchNorm
    ↓
Dense(128, relu, L2) → Dropout(0.3)
    ↓
Dense(3, softmax)
```

**Transfer Learning Strategy:**
- **Stage 1**: Train custom head (frozen base) - 15 epochs, LR: 1e-4
- **Stage 2**: Fine-tune top 30 layers - 25 epochs, LR: 1e-5
- Total Parameters: ~3.5M
- Trainable Parameters: ~1.8M

**Key Advantages:**
- Lightweight and efficient
- Suitable for mobile/edge deployment
- Faster inference time compared to ResNet50

**Preprocessing:**
- Uses MobileNetV2-specific preprocessing
- Scales inputs to [-1, 1] range

---

### 4️⃣ VGG16 (Visual Geometry Group)

**Architecture Overview:**
```
VGG16 (ImageNet Pretrained)
    ↓
GlobalAveragePooling2D
    ↓
BatchNormalization
    ↓
Dense(512, relu, L2) → Dropout(0.5) → BatchNorm
    ↓
Dense(256, relu, L2) → Dropout(0.4) → BatchNorm
    ↓
Dense(128, relu, L2) → Dropout(0.3)
    ↓
Dense(3, softmax)
```

**Three-Stage Training Strategy:**
- **Stage 1**: Train custom head (frozen base) - 20 epochs, LR: 1e-4
- **Stage 2**: Fine-tune Block5 - 25 epochs, LR: 5e-5
- **Stage 3**: Fine-tune Block4+5 - 20 epochs, LR: 1e-5
- Total Parameters: ~16.8M
- Trainable Parameters: ~9.2M

**VGG16 Architecture Details:**
- 5 convolutional blocks (Block1-5)
- Gradual unfreezing for optimal fine-tuning
- Deep architecture with strong feature extraction

**Preprocessing:**
- Uses VGG16-specific preprocessing
- Mean subtraction based on ImageNet statistics

---

## 📊 Performance Comparison

### 🏆 Overall Accuracy Comparison

| Model | Accuracy | Precision | Recall | F1-Score | Parameters | Training Time |
|-------|----------|-----------|--------|----------|------------|---------------|
| **VGG16** | **87.0%** | **87%** | **85%** | **84%** | 16.8M | ~65 epochs |
| **CNN** | **79.0%** | **80%** | **76%** | **75%** | 3.5M | ~30 epochs |
| **ResNet50** | **73.0%** | **72%** | **63%** | **65%** | 25.6M | ~40 epochs |
| **MobileNetV2** | **70.0%** | **69%** | **63%** | **63%** | 3.5M | ~40 epochs |

### 📈 Performance Visualization
```
Overall Accuracy
VGG16        ████████████████████████████████████████████  87%
CNN          ███████████████████████████████████████      79%
ResNet50     ████████████████████████████████████         73%
MobileNetV2  ███████████████████████████████              70%
```

---

## 🎯 Detailed Per-Class Performance

### 1. Angry Emotion

| Model | Precision | Recall | F1-Score | Accuracy |
|-------|-----------|--------|----------|----------|
| **VGG16** | **0.77** | **0.82** | **0.80** | **82.4%** |
| **CNN** | **0.60** | **0.73** | **0.66** | **73.3%** |
| **ResNet50** | **0.53** | **0.42** | **0.47** | **42.4%** |
| **MobileNetV2** | **0.51** | **0.45** | **0.48** | **45.4%** |

**Winner: VGG16** ✅
- Highest recall (82%) - best at detecting angry emotions
- Strong confidence gap (0.115) between correct and incorrect predictions

---

### 2. Sad Emotion

| Model | Precision | Recall | F1-Score | Accuracy |
|-------|-----------|--------|----------|----------|
| **VGG16** | **0.92** | **0.89** | **0.91** | **89.4%** |
| **CNN** | **0.87** | **0.84** | **0.85** | **83.7%** |
| **ResNet50** | **0.76** | **0.88** | **0.82** | **87.5%** |
| **MobileNetV2** | **0.77** | **0.80** | **0.79** | **80.4%** |

**Winner: VGG16** ✅
- Exceptional precision (92%) and F1-score (91%)
- Most reliable for sad emotion detection
- Highest confidence gap (0.258)

---

### 3. Surprise Emotion

| Model | Precision | Recall | F1-Score | Accuracy |
|-------|-----------|--------|----------|----------|
| **VGG16** | **0.82** | **0.84** | **0.83** | **83.7%** |
| **CNN** | **0.78** | **0.70** | **0.74** | **69.5%** |
| **ResNet50** | **0.79** | **0.59** | **0.67** | **58.5%** |
| **MobileNetV2** | **0.63** | **0.63** | **0.63** | **62.6%** |

**Winner: VGG16** ✅
- Balanced precision and recall
- Strong performance across all metrics

---

## 🔍 Error Analysis

### VGG16 (Best Model) - Detailed Analysis

#### Confusion Patterns

| True Label | Predicted | Error Count | Percentage | Avg Confidence |
|------------|-----------|-------------|------------|----------------|
| Sad | Angry | 50 | 60.2% of Sad errors | 0.680 |
| Angry | Sad | 35 | 76.1% of Angry errors | 0.756 |
| Sad | Surprise | 33 | 39.8% of Sad errors | 0.645 |
| Surprise | Sad | 26 | 65.0% of Surprise errors | 0.757 |
| Surprise | Angry | 14 | 35.0% of Surprise errors | 0.652 |
| Angry | Surprise | 11 | 23.9% of Angry errors | 0.661 |

#### Key Insights:
1. **Sad ↔ Angry** is the most common confusion (85 total errors)
2. **Surprise → Sad** is the second most common (26 errors)
3. Model shows good confidence discrimination:
   - Correct predictions: **0.905 avg confidence**
   - Incorrect predictions: **0.697 avg confidence**
   - **Confidence Gap: 0.208** (strong reliability indicator)

#### Error Distribution by Confidence Level

| Confidence Level | Error Count | Percentage |
|------------------|-------------|------------|
| Low (<0.6) | 42 | 24.9% |
| Medium (0.6-0.8) | 58 | 34.3% |
| High (≥0.8) | 69 | 40.8% |

⚠️ **Important Finding**: 40.8% of errors occur with high confidence, indicating systematic misclassification patterns rather than model uncertainty.

---

### CNN Model - Error Analysis

#### Confusion Matrix Highlights:
- **Angry → Sad**: 63 errors (90% of Angry errors)
- **Sad → Angry**: 87 errors (68% of Sad errors)
- **Surprise → Angry**: 40 errors (53.3% of Surprise errors)

#### Confidence Analysis:
- Correct predictions: **0.871 confidence**
- Incorrect predictions: **0.708 confidence**
- Confidence Gap: **0.163**

**Observation**: Good confidence discrimination, but lower overall accuracy than VGG16.

---

### ResNet50 - Error Analysis

#### Major Issues:
1. **Poor Angry Detection**: Only 42.4% recall
   - 145 Angry images misclassified as Sad (96% of errors)
   - **Negative confidence gap** (-0.005): Model equally confident when wrong!

2. **Sad Classification**: Strong (87.5% recall)
   - Best performing class for this model

3. **Surprise Detection**: Moderate (58.5% recall)
   - 69 misclassified as Sad (67.6% of errors)

**Critical Finding**: ResNet50 shows **class bias** toward Sad emotion.

---

### MobileNetV2 - Error Analysis

#### Confusion Patterns:
- **Angry → Sad**: 114 errors (79.7% of Angry errors)
- **Sad → Angry**: 92 errors (59.7% of Sad errors)
- **Surprise → Sad**: 71 errors (77.2% of Surprise errors)

#### Confidence Analysis:
- Correct: **0.753**
- Incorrect: **0.626**
- Gap: **0.127** (lowest among all models)

**Observation**: Model shows uncertainty, reflected in lower confidence scores.

---

## 📊 Comprehensive Model Comparison

### Strengths & Weaknesses

| Model | ✅ Strengths | ⚠️ Weaknesses | 💡 Best Use Case |
|-------|-------------|---------------|------------------|
| **VGG16** | • Highest accuracy (87%)<br>• Excellent across all classes<br>• Strong confidence discrimination<br>• Best angry/surprise detection | • Largest model size (16.8M)<br>• Slower inference<br>• Requires more resources | Production environments where accuracy is critical |
| **CNN** | • Good balance (79%)<br>• Fast training<br>• Smaller model size<br>• No pretrained dependency | • Lower accuracy than VGG16<br>• Moderate angry detection<br>• More training needed | Resource-constrained environments, embedded systems |
| **ResNet50** | • Strong sad detection (88%)<br>• Transfer learning benefits<br>• Proven architecture | • Poor angry detection (42%)<br>• Class bias issues<br>• Largest parameter count | Scenarios where sad emotion is priority |
| **MobileNetV2** | • Most efficient (3.5M params)<br>• Fast inference<br>• Mobile-ready<br>• Lightweight | • Lowest overall accuracy (70%)<br>• Low confidence scores<br>• Needs more tuning | Mobile apps, edge devices, real-time processing |

---

## 🎨 Key Features

### 🖼️ Single Image Prediction
- Upload any facial image
- Get instant emotion prediction
- View confidence scores for all classes
- Visualize prediction confidence

### 📁 Batch Prediction
- Upload multiple images at once
- Process entire folders
- Export results to CSV
- Statistical summary of predictions
- Batch visualization

### 📊 Model Comparison
- Side-by-side performance metrics
- Interactive visualization
- Real-time model switching
- Confidence score comparison

### 📈 Advanced Analytics
- Confusion matrices
- Per-class performance metrics
- Confidence distribution analysis
- Error pattern visualization

---

## 🚀 Installation

### Prerequisites
- Python 3.9 or higher
- pip or PDM package manager
- 4GB+ RAM recommended
- GPU optional (but recommended for training)

### Step 1: Clone Repository
```bash
git clone https://github.com/yourusername/emotion-recognition.git
cd emotion-recognition
```

### Step 2: Install Dependencies

**Using PDM (Recommended):**
```bash
# Install PDM if not already installed
pip install pdm

# Install project dependencies
pdm install
```

**Using pip:**
```bash
pip install -r requirements.txt
```

### Step 3: Download Models
Models are included in the repository under `/model` directory:
- `/model/cnn/` - Custom CNN model
- `/model/resnet/` - ResNet50 model
- `/model/mobilenet/` - MobileNetV2 model
- `/model/vgg/` - VGG16 model

---

## 💻 Usage

### Running Locally

#### Using PDM:
```bash
pdm run streamlit run src/app.py
```

#### Using Python directly:
```bash
streamlit run src/app.py
```

The application will open in your browser at `http://localhost:8501`

### Using the Dashboard

#### 1. Select Model
- Choose from 4 available models in the sidebar
- View model specifications and performance metrics

#### 2. Single Image Prediction
```python
# Select "Single Image" mode
# Upload an image (JPG, PNG, JPEG)
# Click "Predict Emotion"
# View results with confidence scores
```

#### 3. Batch Prediction
```python
# Select "Batch Images" mode
# Upload multiple images
# Process all images at once
# Download results as CSV
```

### Example Code Usage
```python
from tensorflow.keras.models import load_model
from utils.preprocessor import ImagePreprocessor
from PIL import Image

# Load model
model = load_model('model/vgg/vgg16_model.keras')

# Initialize preprocessor
preprocessor = ImagePreprocessor()

# Load and preprocess image
image = Image.open('path/to/image.jpg')
processed_image = preprocessor.preprocess_image(image, 'VGG16')

# Make prediction
prediction = model.predict(processed_image)
emotion = ['Angry', 'Sad', 'Surprise'][prediction.argmax()]
confidence = prediction.max()

print(f"Emotion: {emotion}")
print(f"Confidence: {confidence:.2%}")
```

---

## 🌐 Live Demo

### 🎉 Try it Now!
Access the live demo at: **[Emotion Recognition Dashboard](https://modul6ml-uaphaidar-dimas-heryanto.streamlit.app/)**

### Demo Features:
- ✅ All 4 models available
- ✅ Single & batch image prediction
- ✅ Real-time results
- ✅ Interactive visualizations
- ✅ No installation required

### Sample Images:
You can test the system with the validation dataset or your own images!

---

## 📁 Project Structure
```
emotion-recognition/
│
├── model/                          # Trained models
│   ├── cnn/                       # Custom CNN
│   │   ├── cnn_model.keras
│   │   ├── cnn_model_config.json
│   │   ├── cnn_model_metrics.json
│   │   └── ...
│   ├── resnet/                    # ResNet50
│   │   ├── resnet50_model.keras
│   │   └── ...
│   ├── mobilenet/                 # MobileNetV2
│   │   ├── mobilenetv2_model.keras
│   │   └── ...
│   └── vgg/                       # VGG16
│       ├── vgg16_model.keras
│       └── ...
│
├── src/                           # Source code
│   ├── app.py                     # Main Streamlit app
│   ├── utils/                     # Utility modules
│   │   ├── __init__.py
│   │   ├── model_loader.py        # Model loading logic
│   │   └── preprocessor.py        # Image preprocessing
│   └── pages/                     # UI pages
│       ├── __init__.py
│       ├── home.py
│       ├── single_prediction.py
│       └── batch_prediction.py
│
├── notebooks/                     # Jupyter notebooks
│   ├── training_cnn.ipynb
│   ├── training_resnet50.ipynb
│   ├── training_mobilenetv2.ipynb
│   └── training_vgg16.ipynb
│
├── assets/                        # Static assets
│   ├── images/
│   └── styles.css
│
├── pyproject.toml                 # PDM configuration
├── requirements.txt               # Pip requirements
├── README.md                      # This file
└── .gitignore
```

---

## 🛠️ Technologies Used

### Deep Learning Frameworks
- ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-orange?logo=tensorflow)
- ![Keras](https://img.shields.io/badge/Keras-API-red?logo=keras)

### Web Framework
- ![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red?logo=streamlit)

### Data Processing
- ![NumPy](https://img.shields.io/badge/NumPy-1.24+-blue?logo=numpy)
- ![Pandas](https://img.shields.io/badge/Pandas-2.0+-darkblue?logo=pandas)
- ![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green?logo=opencv)

### Visualization
- ![Plotly](https://img.shields.io/badge/Plotly-5.17+-purple?logo=plotly)
- ![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7+-blue)
- ![Seaborn](https://img.shields.io/badge/Seaborn-0.12+-lightblue)

### Package Management
- ![PDM](https://img.shields.io/badge/PDM-Package_Manager-blueviolet)

---

## 📈 Future Improvements

### 🔮 Planned Features
1. **Additional Emotions**: Expand to 7 emotions (Happy, Fear, Disgust, Neutral)
2. **Real-time Video**: Webcam integration for live emotion detection
3. **Model Ensemble**: Combine predictions from multiple models
4. **API Development**: RESTful API for integration
5. **Mobile App**: Flutter/React Native implementation
6. **Edge Deployment**: TensorFlow Lite conversion

### 🔬 Research Directions
- Attention mechanisms for better feature extraction
- Self-supervised learning approaches
- Cross-dataset generalization
- Adversarial training for robustness

---

## 📚 References

### Dataset
- Kapadnis, S. (2023). *Emotion Recognition Dataset*. Kaggle. [Link](https://www.kaggle.com/datasets/sujaykapadnis/emotion-recognition-dataset)

### Model Architectures
- He, K., et al. (2016). *Deep Residual Learning for Image Recognition*. CVPR.
- Sandler, M., et al. (2018). *MobileNetV2: Inverted Residuals and Linear Bottlenecks*. CVPR.
- Simonyan, K., & Zisserman, A. (2014). *Very Deep Convolutional Networks*. ICLR.

### Tools & Frameworks
- Abadi, M., et al. (2016). *TensorFlow: Large-Scale Machine Learning*. OSDI.
- Streamlit Inc. (2023). *Streamlit Documentation*. [Link](https://docs.streamlit.io/)

---

## 👥 Contributors

### Project Team
- **[Haidar Dimas Heryanto]** - Lead Developer & ML Engineer
  - Model training and optimization
  - Web application development
  - Documentation

### Academic Supervisor
- **[Supervisor Name]** - Course Instructor
  - Project guidance
  - Technical consultation

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Kaggle** for providing the emotion recognition dataset
- **TensorFlow** and **Keras** teams for the excellent deep learning framework
- **Streamlit** for the intuitive web framework
- **Open Source Community** for pre-trained models and resources

---

## 📞 Contact

### Project Links
- **Live Demo**: [Streamlit Cloud](https://modul6ml-uaphaidar-dimas-heryanto.streamlit.app/)
- **Repository**: [GitHub](https://github.com/Dimashery/Modul_6_ML-UAP_Haidar-Dimas-Heryanto)
- **Dataset**: [Kaggle](https://www.kaggle.com/datasets/sujaykapadnis/emotion-recognition-dataset)

### Get in Touch
- 📧 Email: haidardimas003@gmail.com
- 💼 LinkedIn: -
- 🐱 GitHub: [@Dimashery](https://github.com/Dimashery)

---

<div align="center">
  
### ⭐ Star this repository if you found it helpful!

**Made with ❤️ for Machine Learning Course - Semester 7**

</div>

---

## 📊 Quick Stats

<div align="center">

| Metric | Value |
|--------|-------|
| 🎯 Best Accuracy | 87.0% (VGG16) |
| 📦 Models Trained | 4 |
| 🎭 Emotions Detected | 3 |
| 📸 Test Images | 1,294 |
| ⚡ Inference Time | <100ms |
| 🏆 Best F1-Score | 0.84 (VGG16) |

</div>

---

**Last Updated**: December 2024 | **Version**: 1.0.0
