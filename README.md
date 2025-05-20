# Fish Image Classification Using Deep Learning

## Overview
This project focuses on classifying **fish images** into multiple categories using **deep learning models**. The task involves:
- Training a **CNN from scratch**.
- Leveraging **transfer learning** with **pre-trained models**.
- Saving trained models for future use.
- Deploying a **Streamlit application** that predicts fish categories from user-uploaded images.

## Business Use Cases
**Enhanced Accuracy** – Determine the best model architecture for fish image classification.  
**Deployment Ready** – Create a **user-friendly web application** for real-time predictions.  
**Model Comparison** – Evaluate multiple models to select the best approach.  

## Approach
### **Data Preprocessing & Augmentation**
- Rescale images to '[0,1]' range.
- Apply **augmentation** techniques:
  - Rotation
  - Zoom
  - Flipping
  - Random cropping

### **Model Training**
1. Train a **CNN model from scratch**.
2. Experiment with **five pre-trained models**:
   - VGG16
   - ResNet50
   - MobileNet
   - InceptionV3
   - EfficientNetB0
3. **Fine-tune** models using the fish dataset.
4. **Save the best model** The models are saved in .h5 format for future use.

### **Model Evaluation**
- Compare models based on:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - Confusion Matrix
- **Visualize** training history (accuracy & loss curves).

### **Deployment**
A **Streamlit web app** with features:
- **Image Upload** – Users upload fish images.
- **Prediction** – Displays fish category.
- **Confidence Score** – Shows model certainty.
