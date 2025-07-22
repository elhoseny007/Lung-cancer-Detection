# Lung Cancer Detection using Transfer Learning (VGG16)

This project uses transfer learning with the VGG16 model to detect lung cancer from CT scan images. The approach combines image preprocessing, data augmentation, and a custom classifier head on top of the VGG16 base model.

## 🧠 Project Objective

Early detection of lung cancer can improve treatment outcomes. This project aims to classify CT images into two classes:
- Cancer
- Non-Cancer

## 🗂️ Dataset

- The dataset contains labeled CT scan images.
- Images were converted to RGB, sharpened, and augmented before training.

## 🖼️ Preprocessing Steps

- Convert grayscale images to **RGB**.
- Apply **image sharpening** using a sharpening kernel.
- Resize all images to **224x224** (VGG16 input size).
- Perform **data augmentation**: rotation, zoom, horizontal/vertical shifts.

## 🧰 Model Architecture

- **Base Model**: VGG16 (pretrained on ImageNet, base layers frozen).
- **Classifier Head**:
  - GlobalAveragePooling2D
  - Dense(128, activation='relu')
  - Dropout(0.5)
  - Dense(1, activation='sigmoid')

## ⚙️ Requirements

```bash
tensorflow
collections
os
keras
matplotlib
opencv-python
numpy
pandas
seaborn
