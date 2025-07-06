# 🌿 Plant Disease Prediction System from Leaf Images

This project implements a deep learning-based system to detect plant diseases using a custom Convolutional Neural Network (CNN) built from scratch with TensorFlow and Keras. The model is trained on the PlantVillage dataset and classifies 38 different types of plant diseases with high accuracy.

---

## 📌 Overview

Agriculture is the backbone of many economies, but plant diseases can cause massive losses in crop yield. This system aims to provide a reliable, fast, and lightweight solution to identify diseases from leaf images, enabling early intervention and treatment.

---

## 🗂️ Dataset

- **Dataset**: [PlantVillage](https://www.kaggle.com/datasets/emmarex/plantdisease)
- **Total Classes**: 38
- **Structure**: Images are categorized into folders by disease and crop type.
- **Split**:
  - 70% for training
  - 20% of training used for validation
  - 30% for testing

---

## 🧠 Model Summary

- Custom CNN built from scratch (no pretrained models)
- Architecture:
  - 4 Conv2D + ReLU + MaxPooling2D blocks
  - Flatten → Dense layers
  - Dropout for regularization
  - Final Dense layer with softmax activation
- Optimizer: Adam
- Loss: Categorical Crossentropy
- Metrics: Accuracy, Precision, Recall, F1-score

---

## 🔧 Technologies Used

- Python 3.x
- TensorFlow / Keras
- NumPy, Pandas
- Matplotlib, Seaborn
- Scikit-learn
- OpenCV (for preprocessing)
- Google Colab (for training)

---

## 🚀 How to Use

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/plant-disease-prediction.git
cd plant-disease-prediction
