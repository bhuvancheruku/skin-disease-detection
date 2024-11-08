# Skin Disease Detection

This repository hosts a **Streamlit web application** for detecting skin diseases using **image processing techniques** and **machine learning**. The app takes an uploaded image, processes it to extract relevant features, and uses a trained machine learning model to classify the skin condition as normal or abnormal.

## Features
- **Image Upload**: Allows users to upload an image of a skin condition for analysis.
- **Image Preprocessing**: Converts images to grayscale, applies Gaussian blurring, and extracts features using techniques such as GLCM (Gray Level Co-occurrence Matrix).
- **Machine Learning Prediction**: A trained Support Vector Machine (SVM) model predicts whether the skin is in a normal or abnormal state.

## Project Structure
The project is organized as follows:
- `app.py`: The main Streamlit application that manages the user interface and displays results.
- `image_processing.py`: Contains all functions for image preprocessing, feature extraction, and model prediction.
- `requirements.txt`: Lists all dependencies for setting up the project environment.
- `README.md`: Project documentation.

## Prerequisites
To run this application, you need:
- Python 3.8 or later
- Required Python packages (listed in `requirements.txt`)
