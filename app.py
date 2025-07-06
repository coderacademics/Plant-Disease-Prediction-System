import streamlit as st
import tensorflow as tf
import numpy as np
import keras

class_names = [
    "Apple - Apple scab", "Apple - Black rot", "Apple - Cedar apple rust", "Apple - Healthy",
    "Blueberry - Healthy", "Cherry - Powdery mildew", "Cherry - Healthy",
    "Corn - Cercospora leaf spot (Gray leaf spot)", "Corn - Common rust", "Corn - Northern Leaf Blight",
    "Corn - Healthy", "Grape - Black rot", "Grape - Esca (Black Measles)", "Grape - Leaf blight",
    "Grape - Healthy", "Orange - Citrus greening", "Peach - Bacterial spot", "Peach - Healthy",
    "Pepper bell - Bacterial spot", "Pepper bell - Healthy", "Potato - Early blight", "Potato - Late blight",
    "Potato - Healthy", "Raspberry - Healthy", "Soybean - Healthy", "Squash - Powdery mildew",
    "Strawberry - Leaf scorch", "Strawberry - Healthy", "Tomato - Bacterial spot", "Tomato - Early blight",
    "Tomato - Late blight", "Tomato - Leaf Mold", "Tomato - Septoria leaf spot", 
    "Tomato - Spider mites (Two-spotted)", "Tomato - Target Spot", 
    "Tomato - Tomato Yellow Leaf Curl Virus", "Tomato - Tomato mosaic virus", "Tomato - Healthy"
]


# Tensorflow model prediction
def model_prediction():
    model = tf.keras.model.load_model("trained_model_v3.keras")
    image = tf.keras.preprocessing.image.load_img(image_path,size=[128,128])
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    prediction = model.predict(input_arr) 
    result = np.argmax(prediction)
    return result


st.sidebar.title("DashBoard")

curr_page = st.sidebar.selectbox("Go to :",["Home","About Us","Predict Disease"])

if curr_page == "Home":
    st.header("PLANT LEAF DISEASE PREDICTION SYSTEM")
    image = "home_page_bg.jpg"
    st.image(image,use_container_width=True)
    st.markdown("""
    Welcome to the Plant Leaf Disease Prediction System! 🌿🔍
    
    Our mission is to help in identifying plant diseases efficiently using the image of their leaves. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. 
    Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. **Upload Image:** Go to the **Predict Disease** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using machine learning algorith and give result of the predicted disease.
    3. **Results:** View the results and recommendations for further action to eliminate this.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes lightweight and fast machine learning model for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Predict Diease** page in the sidebar to upload an image and experience the power of our Plant Leaf Disease Prediction System!

    ### About Us
    Plaase visit the About page to learn more about the internal working of the project.
    """)
elif curr_page == "About Us":
    st.header("About the project")
    st.markdown("""
    Welcome to my **Plant Leaf Disease Prediction System**, a research-oriented web application developed as part of my MCA final year project.

    This system utilizes a **custom-built Convolutional Neural Network (CNN)** created from scratch using **TensorFlow and Keras**, capable of accurately classifying **38 different categories** of plant leaf diseases, including healthy leaf conditions. The model was trained on the publicly available **PlantVillage dataset** and optimized for both accuracy and lightweight performance.

    This is not just an academic initiative — the entire project has been submitted to a **Springer Nature** research journal, aiming to contribute to the field of **AI in Agriculture** by demonstrating the potential of deep learning in smart crop disease detection.

    Feel free to explore the features of this system through the sidebar, test the prediction module, and learn how deep learning can make a real-world impact in modern agriculture.
    👉 Want to try the model?
    """)
    
    if st.button("Go to Predict Disease Page"):
        st.session_state["curr_page"] = "Predict Disease"
else:
    st.header("PREDICT THE DISEASE")
    uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Leaf Image", use_container_width=True)

        # Load and preprocess image
        image = tf.keras.preprocessing.image.load_img(uploaded_file, target_size=(128, 128))
        input_arr = tf.keras.preprocessing.image.img_to_array(image)
        input_arr = np.expand_dims(input_arr, axis=0)  # Add batch dimension

        # Load model (load once for efficiency in real case)
        model = tf.keras.models.load_model("trained_model_v2.0.keras")

        # Predict
        prediction = model.predict(input_arr)
        predicted_class = np.argmax(prediction)
        class_name = class_names[predicted_class]

        st.success(f"Predicted Class: **{class_name}**")