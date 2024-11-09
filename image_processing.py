import os
import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops

def load_image(image_id, image_paths):
    """Load an image from either of the image folders."""
    for folder in image_paths:
        img_path = os.path.join(folder, f"{image_id}.jpg")
        if os.path.exists(img_path):
            return cv2.imread(img_path)
    return None  # Return None if image is not found

def preprocess_image(image, target_size=(128, 128)):
    """Convert image to grayscale, resize, and apply Gaussian blur."""
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized_image = cv2.resize(gray_image, target_size)
    blurred_image = cv2.GaussianBlur(resized_image, (5, 5), 0)
    normalized_image = blurred_image / 255.0
    return normalized_image

def extract_features(image):
    """Extract texture features using GLCM."""
    glcm = graycomatrix((image * 255).astype('uint8'), distances=[1], angles=[0], symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    asm = graycoprops(glcm, 'ASM')[0, 0]
    
    features = np.array([contrast, energy, homogeneity, dissimilarity, correlation, asm])
    normalized_features = features / np.max(features)  # Normalize each feature
    return normalized_features

def process_row(row, image_paths):
    """Process a single row: load image, preprocess, and extract features."""
    img = load_image(row['image_id'], image_paths)
    if img is not None:
        processed_img = preprocess_image(img)
        features = extract_features(processed_img)
        return features, row['label']
    return None, None

def process_image_data(metadata, image_paths):
    """Process images sequentially without multiprocessing."""
    X = []
    y = []
    for _, row in metadata.iterrows():
        features, label = process_row(row, image_paths)
        if features is not None:
            X.append(features)
            y.append(label)
    
    return np.array(X), np.array(y)
