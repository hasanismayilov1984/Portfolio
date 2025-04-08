import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Load your saved model
model = tf.keras.models.load_model("animal_classifier_model.h5")

# Set class names (same order as in training)
class_names = ['butterfly', 'cat', 'chicken', 'cow', 'dog', 'elephant', 'horse', 'sheep', 'spider', 'squirrel']

# App title
st.title("🐾 Animal Image Classifier")
st.write("Upload an image of an animal and the model will tell you what it is.")

# Image upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess image
    img = image.resize((180, 180))
    img_array = np.array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Prediction
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]

    st.markdown(f"### 🧠 Prediction: **{predicted_class.upper()}**")
