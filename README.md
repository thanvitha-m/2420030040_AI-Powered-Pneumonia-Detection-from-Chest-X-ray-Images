# Deep Learning Framework for Automated Pneumonia Detection Using Chest X-ray Images

## 📌 Project Overview

Pneumonia is a serious respiratory condition that can be identified from chest X-ray (CXR) images. Manual interpretation of chest X-rays can be time-consuming and may vary depending on workload and experience.

This project develops a **deep-learning-based research prototype for automated pneumonia classification from chest X-ray images**.

The current implementation focuses on addressing the research gap of **limited explainability** in pneumonia detection systems. The system combines:

- **DenseNet-121** for pneumonia classification
- **Transfer Learning** for efficient model training
- **Grad-CAM** for visual model interpretability
- **FastAPI** for an interactive web-based prediction interface
- **Performance evaluation** using multiple classification metrics

> **Important:** This project is an academic research prototype. Its predictions and Grad-CAM visualizations are intended for research and demonstration purposes and **must not be considered a medical diagnosis**.

---

## 🎯 Objectives

The major objectives of the project are:

1. Develop an automated pneumonia classification system using chest X-ray images.
2. Apply **deep learning and transfer learning** for image classification.
3. Address the research gap of **limited explainability** in pneumonia detection.
4. Use **Grad-CAM** to visualize image regions that contributed to the model prediction.
5. Evaluate the model using multiple performance metrics.
6. Develop a simple **FastAPI-based web interface** for uploading an X-ray and viewing the prediction and Grad-CAM explanation.
7. Provide a foundation for future extension to multiple datasets, architectures, and natural-language explanations.

---

# 🔬 Research Gap

Based on the literature reviewed for this project, several existing pneumonia detection approaches focus primarily on classification performance while providing limited or predominantly visual explanations.

The selected research gap for the current implementation is:

> **Limited Explainability in Automated Pneumonia Detection**

Many deep-learning models can classify a chest X-ray as Normal or Pneumonia, but understanding **which image regions contributed to the prediction** is important when developing an interpretable research system.

### Our approach

To address this gap, the current implementation integrates:

**DenseNet-121 → Prediction → Grad-CAM → Visual Explanation**

Grad-CAM is an established explainability technique. Therefore, the contribution of this project is **not the invention of Grad-CAM**, but its integration into the pneumonia-classification workflow together with model evaluation and a web-based demonstration.

---

# 🏗️ System Architecture

The implemented system follows the pipeline:

```text
Chest X-ray Image
        ↓
Data Preprocessing
        ↓
Training / Validation / Testing
        ↓
DenseNet-121 Transfer Learning
        ↓
Normal / Pneumonia Prediction
        ↓
Performance Evaluation
        ↓
Grad-CAM Explainability
        ↓
FastAPI Web Application
        ↓
Prediction + Confidence + Visualization