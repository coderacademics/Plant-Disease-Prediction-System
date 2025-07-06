# 🌿 Plant Disease Prediction System from Leaf Images

A deep learning-based plant disease detection system that identifies diseases from leaf images using a custom CNN model built from scratch with TensorFlow and Keras. The model is trained on the PlantVillage dataset and classifies 38 different plant disease classes with high accuracy.

---

## 📌 Overview

Plant diseases impact crop productivity and food security. This project builds a lightweight, custom CNN model to classify plant diseases using leaf images. The model is trained using a Jupyter notebook and can be used via a Streamlit web app interface.

---

## 🗂️ Project Structure

```
├── app.py                    # Streamlit web app
├── notebook.ipynb           # Jupyter notebook with full training pipeline
├── trained_model_v2.0.keras # Saved trained model (custom CNN)
├── requirements.txt         # Dependencies
├── readme.md                # Project documentation
├── home_page_bg.jpg         # Background image for web app
├── plantapp/                # (Optional) Streamlit submodule
```

---

## 📊 Dataset

- **Source**: [PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease)
- **Total Classes**: 38
- **Splits**:
  - 70% for training (with 20% used for validation)
  - 30% for testing
- **Preprocessing**:
  - Image resizing
  - Normalization
  - Data augmentation
  - (Optional) GLCM feature extraction

---

## 🧠 Model Details

- **Architecture**: 4 Conv2D layers + ReLU + MaxPooling2D  
- **Regularization**: Dropout  
- **Final Layer**: Dense with softmax  
- **Optimizer**: Adam  
- **Loss Function**: Categorical Crossentropy  
- **Evaluation**: Accuracy, Precision, Recall, F1-score

---

## 🛠️ Technologies Used

- Python 3.x  
- TensorFlow / Keras  
- NumPy, Pandas  
- OpenCV (for preprocessing)  
- Matplotlib, Seaborn  
- Scikit-learn  
- Streamlit (for web interface)

---

## 🚀 How to Use

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/plant-disease-prediction.git
cd plant-disease-prediction
```

### 2️⃣ Install Required Libraries

```bash
pip install -r requirements.txt
```

### 3️⃣ Train the Model (optional if using pre-trained)

Open `notebook.ipynb` and run all cells to:
- Load the dataset
- Preprocess & augment
- Train the CNN model
- Evaluate metrics

This will also generate the `trained_model_v2.0.keras`.

### 4️⃣ Launch the Streamlit Web App

```bash
streamlit run app.py
```

> Make sure `trained_model_v2.0.keras` and `home_page_bg.jpg` are in the same folder as `app.py`.

---

## 📊 Model Performance

- **Test Accuracy**: ~97–98%
- **Evaluation Metrics**:
  - Confusion Matrix
  - Precision, Recall, F1-Score
  - Epoch-wise training/validation plots

---

## 🌱 Applications

- Smart farming apps  
- Disease monitoring in agriculture  
- Field-level disease alerts through mobile/web interface

---

## 🔮 Future Work

- Integrate webcam/live camera prediction in the app  
- Deploy the web app with multilingual support (Hindi, Bengali)  
- Use semi-supervised learning with real-field drone images  
- Add attention mechanisms or transformer-based enhancements  
- Export model to ONNX or TensorFlow Lite for mobile deployment  

---

## 👨‍💻 Author

**Saikat Mohanta**  
📫 [LinkedIn](https://www.linkedin.com/in/saikat-mohanta43434/)  
📧 saikatmohanta43434@gmail.com

---

## 📜 License

This project is licensed under the **MIT License**.  
Feel free to fork, modify, or use it for your academic or personal projects.