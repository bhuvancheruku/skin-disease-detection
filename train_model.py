import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report
import joblib
from imblearn.over_sampling import SMOTE # type: ignore
import sys 
sys.path.append(r"C:\Users\thiru\OneDrive\Desktop\myproject\skin_disease_detection")
from image_processing import process_image_data  # type: ignore # Ensure process_image_data handles images without multiprocessing

# Paths to dataset and image folders
DATASET_PATH = r"C:\Users\thiru\OneDrive\Desktop\myproject\dataset"
IMAGE_PATH_1 = os.path.join(DATASET_PATH, "HAM10000_images_part_1")
IMAGE_PATH_2 = os.path.join(DATASET_PATH, "HAM10000_images_part_2")
METADATA_FILE = os.path.join(DATASET_PATH, "HAM10000_metadata.csv")

print("Loading metadata...")
# Load metadata
metadata = pd.read_csv(METADATA_FILE)
disease_labels = metadata['dx'].unique()
class_to_label = {name: idx for idx, name in enumerate(disease_labels)}
label_to_class = {v: k for k, v in class_to_label.items()}
metadata['label'] = metadata['dx'].map(class_to_label)
print(f"Metadata loaded. Found {len(metadata)} entries.")

print("Processing images and extracting features...")
# Process images and extract features
X, y = process_image_data(metadata, [IMAGE_PATH_1, IMAGE_PATH_2])
print(f"Feature extraction complete. Extracted {X.shape[0]} samples with {X.shape[1]} features each.")

print("Handling class imbalance with SMOTE...")
# Handle class imbalance using SMOTE
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
print(f"Class imbalance handled. Dataset now has {X_resampled.shape[0]} samples.")

print("Splitting dataset into training and testing sets...")
# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
print(f"Data split complete. Training set has {X_train.shape[0]} samples, and testing set has {X_test.shape[0]} samples.")

print("Setting up the SVM model pipeline...")
# Define the SVM model with a pipeline (StandardScaler + SVC)
model = make_pipeline(StandardScaler(), SVC(kernel='linear', probability=True, class_weight='balanced'))

print("Performing hyperparameter tuning with GridSearchCV...")
# Hyperparameter tuning using GridSearchCV without multiprocessing
param_grid = {
    'svc__C': [0.1, 1, 10, 100],  # Regularization parameter
    'svc__kernel': ['linear', 'rbf'],  # Try different kernel types
}
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=1, verbose=3)  # n_jobs=1 disables parallelism
grid_search.fit(X_train, y_train)
print("Grid search complete.")

# Best model after hyperparameter tuning
best_model = grid_search.best_estimator_
print(f"Best parameters found: {grid_search.best_params_}")

print("Evaluating the model on the test set...")
# Evaluate the model
y_pred = best_model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=disease_labels))

# Save the trained model and label mapping
print("Saving the trained model and label mapping...")
joblib.dump(best_model, "skin_disease_model.joblib")
joblib.dump(label_to_class, "label_mapping.joblib")
print("Model and label mapping saved successfully.")
