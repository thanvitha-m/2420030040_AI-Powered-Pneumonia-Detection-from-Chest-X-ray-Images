# AI-Powered Pneumonia Detection from Chest X-ray Images

## **Project Overview**

**AI-Powered Pneumonia Detection from Chest X-ray Images** is an explainable deep-learning framework designed to automatically detect pneumonia from chest X-ray images.

The project compares multiple deep-learning architectures, including **CNN, DenseNet-121, EfficientNet, and Transformer models**, using multiple chest X-ray datasets. To improve the interpretability of predictions, the framework integrates **Grad-CAM** for visual explanations and **BERT + LLM** for natural-language explanations.

The final system is planned as an end-to-end web application using **FastAPI** for the backend and **React.js** for the frontend.

---

## **Problem Statement**

Pneumonia detection from chest X-ray images can be time-consuming and may require expert radiological interpretation. Traditional deep-learning models can provide accurate predictions but often operate as black boxes, making it difficult to understand why a particular prediction was made.

This project aims to develop an **automated and explainable pneumonia detection framework** that combines deep-learning classification with visual and textual explanations.

---

## **Objectives**

- **Develop** an automated pneumonia detection system using chest X-ray images.
- **Compare** CNN, DenseNet-121, EfficientNet, and Transformer architectures.
- **Evaluate** the models using Accuracy, Precision, Recall, F1-score, ROC-AUC, and Confusion Matrix.
- **Improve model generalisation** using multiple chest X-ray datasets.
- **Provide visual explanations** using Grad-CAM.
- **Generate understandable textual explanations** using BERT + LLM.
- **Develop an end-to-end web application** using React.js and FastAPI.

---

## **Datasets**

The project uses three publicly available chest X-ray datasets:

### **1. NIH ChestX-ray14**

A large-scale chest X-ray dataset used for training and evaluation of pneumonia detection models.

### **2. RSNA Pneumonia Detection Challenge**

A chest X-ray dataset containing pneumonia-related annotations used for developing and evaluating automated pneumonia detection systems.

### **3. VinDr-CXR**

A chest X-ray dataset containing expert annotations that can be used for medical image analysis and model evaluation.

---

## **Methodology**

The proposed workflow consists of the following stages:

**Chest X-ray Dataset**  
↓  
**Data Preprocessing**  
↓  
**Image Augmentation**  
↓  
**Model Training**  
↓  
**CNN / DenseNet-121 / EfficientNet / Transformer**  
↓  
**Model Evaluation**  
↓  
**Best Model Selection**  
↓  
**Grad-CAM Explainability**  
↓  
**BERT + LLM Textual Explanation**  
↓  
**FastAPI Backend**  
↓  
**React.js Frontend**  
↓  
**Final AI-Assisted Pneumonia Detection System**

---

## **Deep Learning Models**

### **CNN**

A baseline Convolutional Neural Network is used for pneumonia classification from chest X-ray images.

### **DenseNet-121**

DenseNet-121 is used to learn detailed visual features through dense connections between network layers.

### **EfficientNet**

EfficientNet is evaluated for achieving an effective balance between model performance and computational efficiency.

### **Transformer**

A Transformer-based architecture is explored for image representation and pneumonia classification.

---

## **Explainable AI**

### **Grad-CAM**

**Gradient-weighted Class Activation Mapping (Grad-CAM)** is used to visualize the regions of the chest X-ray that contribute to the model's prediction.

The system produces a heatmap that can be overlaid on the original X-ray image.

**Input X-ray → Model Prediction → Grad-CAM → Visual Heatmap**

---

## **Natural Language Explanation**

The framework incorporates **BERT + LLM** to generate understandable textual explanations based on the model's prediction and associated information.

This provides two forms of explanation:

- **Visual Explanation:** Grad-CAM heatmap
- **Textual Explanation:** BERT + LLM generated summary

---

## **Model Evaluation**

The trained models will be evaluated using:

- **Accuracy**
- **Precision**
- **Recall**
- **F1-score**
- **ROC-AUC**
- **Confusion Matrix**

The performance of CNN, DenseNet-121, EfficientNet, and Transformer models will be compared to identify the most suitable model for the proposed system.

**Actual performance values will be added after model training and experimentation.**

---

## **Technologies Used**

- **Python**
- **PyTorch**
- **CNN / Deep Learning**
- **DenseNet-121 & EfficientNet**
- **Grad-CAM**
- **BERT + LLM**
- **Google Colab with GPU**
- **FastAPI**
- **React.js**

---

## **System Architecture**

```text
                ┌──────────────────────┐
                │   Chest X-ray Image  │
                └──────────┬───────────┘
                           ↓
                ┌──────────────────────┐
                │   Preprocessing      │
                └──────────┬───────────┘
                           ↓
        ┌────────────────────────────────────┐
        │       Deep Learning Models         │
        │                                    │
        │ CNN | DenseNet | EfficientNet      │
        │              | Transformer          │
        └────────────────┬───────────────────┘
                         ↓
                ┌──────────────────────┐
                │     Prediction       │
                └──────────┬───────────┘
                           ↓
             ┌──────────────────────────┐
             │     Grad-CAM             │
             │   Visual Explanation     │
             └────────────┬─────────────┘
                          ↓
             ┌──────────────────────────┐
             │     BERT + LLM           │
             │  Textual Explanation     │
             └────────────┬─────────────┘
                          ↓
             ┌──────────────────────────┐
             │    FastAPI Backend       │
             └────────────┬─────────────┘
                          ↓
             ┌──────────────────────────┐
             │     React Frontend       │
             └──────────────────────────┘
