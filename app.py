import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
import joblib
from PIL import Image

# Load the trained model and label mapping
model = load_model("skin_disease_cnn_model.h5")
label_to_class = joblib.load("label_mapping.joblib")

# Preprocess the uploaded image
def preprocess_image(image):
    """Resize and normalize the uploaded image."""
    image = image.resize((128, 128))
    image = np.array(image) / 255.0
    return np.expand_dims(image, axis=0)

# Custom CSS for bright colors and landing page
st.markdown("""
    <style>
        body {
            background-color: #f5f5f5;
            color: #333;
        }
        .title {
            font-size: 3em;
            font-weight: bold;
            color: #ff6347;
            text-align: center;
            margin-bottom: 20px;
        }
        .navbar {
            background-color: #ff6347;
            color: white;
            padding: 10px;
            text-align: center;
        }
        .navbar a {
            margin: 0 15px;
            color: white;
            text-decoration: none;
            font-weight: bold;
        }
        .navbar a:hover {
            color: #f0f8ff;
        }
        .content {
            margin-top: 30px;
        }
        .button {
            background-color: #ff6347;
            color: white;
            padding: 10px 20px;
            border-radius: 5px;
            font-size: 1.2em;
        }
        .button:hover {
            background-color: #f0f8ff;
        }
    </style>
""", unsafe_allow_html=True)

# Navigation bar
def navbar():
    st.markdown("""
        <div class="navbar">
            <a href="#home" id="home-link">Home</a>
            <a href="#detection" id="detection-link">Skin Disease Detection</a>
        </div>
    """, unsafe_allow_html=True)

# Home page
def home_page():
    st.markdown('<div class="title">Welcome to Skin Disease Detection</div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="content">
            <h2 style="text-align: center;">Revolutionizing Dermatology with AI</h2>
            <p style="font-size: 1.2em; text-align: center;">Our advanced AI model helps you detect skin diseases by analyzing images. Upload a photo of a skin lesion, and our system will provide a diagnosis along with the confidence level.</p>
        </div>
    """, unsafe_allow_html=True)
    
    # "Go to Detection" Button
    if st.button("Go to Detection"):
        st.session_state.page = "Skin Disease Detection"

# Full name to code name mapping for the HAM10000 dataset
class_names = {
    "ak": "Actinic keratoses",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic nevi",
    "vasc": "Vascular lesions"
}

# Skin Disease Detection page
def skin_disease_detection():
    st.title("Skin Disease Detection")
    
    # Patient details input
    patient_name = st.text_input("Enter Patient's Name:")
    patient_age = st.number_input("Enter Patient's Age:", min_value=0, max_value=120, step=1)

    # Image upload
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.write("Classifying...")

        # Preprocess and predict
        img_array = preprocess_image(image)
        predictions = model.predict(img_array)

        # Get the predicted class and accuracy
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        confidence = np.max(predictions) * 100  # Confidence as a percentage

        # Map the predicted class index to its code name
        code_name = label_to_class[predicted_class_index]

        # Get the full name using the code name
        full_name = class_names.get(code_name, "Unknown")

        # Display the results
        st.write(f"**Patient Name:** {patient_name}")
        st.write(f"**Patient Age:** {patient_age}")
        st.write(f"**Predicted Skin Disease:** {full_name} ({code_name})")
        st.write(f"**Confidence Level:** {confidence:.2f}%")

# Initialize session state for page management
if 'page' not in st.session_state:
    st.session_state.page = "Home"

# Add navbar at the top of the page
navbar()

# Page navigation
if st.session_state.page == "Home":
    home_page()
elif st.session_state.page == "Skin Disease Detection":
    skin_disease_detection()
