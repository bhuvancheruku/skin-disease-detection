import cv2
import numpy as np
import os
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from image_processing import preprocess_image, extract_features  # Import functions from image_processing.py

# Path to dataset
NORMAL_DIR = "path/to/normal_images"
ABNORMAL_DIR = "path/to/abnormal_images"

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images

# Load dataset
normal_images = load_images_from_folder(NORMAL_DIR)
abnormal_images = load_images_from_folder(ABNORMAL_DIR)

# Prepare data and labels
X = []
y = []

# Label 0 for normal and 1 for abnormal
for img in normal_images:
    processed_img = preprocess_image(img)
    features = extract_features(processed_img)
    X.append(features)
    y.append(0)

for img in abnormal_images:
    processed_img = preprocess_image(img)
    features = extract_features(processed_img)
    X.append(features)
    y.append(1)

# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an SVM model pipeline
model = make_pipeline(StandardScaler(), SVC(kernel='linear', probability=True))

# Train the model
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Model accuracy:", accuracy)

# Save the model
joblib.dump(model, "skin_disease_model.joblib")
print("Model saved as skin_disease_model.joblib")
