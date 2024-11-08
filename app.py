import streamlit as st
import cv2
import numpy as np
import joblib
from image_processing import preprocess_image, extract_features, load_model, predict_skin_condition

# Load pre-trained model
model = joblib.load("skin_disease_model.joblib")

# Streamlit App
st.title("Skin Disease Detection")

st.write("Upload an image to detect skin disease.")
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess and extract features
    processed_image = preprocess_image(image)
    st.image(processed_image, caption='Processed Image (Grayscale & Blurred)', use_column_width=True)

    features = extract_features(processed_image)
    st.write("Extracted Features:", features)

    # Predict
    prediction = predict_skin_condition(model, features)
    st.write("Predicted Skin Condition:", "Normal" if prediction == 0 else "Abnormal")

st.write("This application uses image processing and machine learning to analyze skin disease.")
