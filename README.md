# CropVision 🌱

AI-powered plant disease classification using transfer learning and explainable AI.

## Overview

CropVision is an image-based plant health classification system that analyzes leaf images and predicts one of five plant health classes.

The project uses a pretrained MobileNetV2 model with transfer learning and provides predictions through a Streamlit web interface.

## Features

- Plant leaf image classification
- MobileNetV2 transfer learning
- Top-3 prediction distribution
- Confidence score
- Grad-CAM explainability
- Streamlit interface
- Model evaluation with confusion matrix
- Training performance visualization

## Classes

CropVision recognizes:

1. Bell Pepper — Bacterial Spot
2. Bell Pepper — Healthy
3. Potato — Early Blight
4. Potato — Late Blight
5. Potato — Healthy

## Model

### Architecture

Input Image
→ Data Augmentation
→ MobileNetV2
→ Global Average Pooling
→ Dropout
→ Dense Softmax Classifier

## Model Performance

Validation accuracy:

**94.52%**

Validation samples:

**821**

The evaluation includes class-wise precision, recall and F1-score.

## Explainability

CropVision uses Grad-CAM to visualize image regions that contributed to the selected prediction.

The visualization is intended to improve model interpretability and should not be treated as independent proof of a disease diagnosis.

## Tech Stack

- Python
- TensorFlow
- Keras
- MobileNetV2
- Streamlit
- NumPy
- Pillow
- Matplotlib
- Scikit-learn

## Project Structure

```text
Cropvision/
├── models/
├── test_images/
├── app.py
├── generate_gradcam.py
├── prepare_dataset.py
├── train_model.py
├── verify_model.py
├── requirements.txt
└── README.md