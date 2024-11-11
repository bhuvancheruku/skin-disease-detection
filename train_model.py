import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import joblib
from image_processing import load_data

# Paths to dataset and image folders
DATASET_PATH = r"D:\projectv2\dataset\HAM10000_metadata.csv"
IMAGE_PATH_1 = r"D:\projectv2\dataset\HAM10000_images_part_1"
IMAGE_PATH_2 = r"D:\projectv2\dataset\HAM10000_images_part_2"

# Load data
print("Loading data...")
X, y, label_to_class = load_data(DATASET_PATH, [IMAGE_PATH_1, IMAGE_PATH_2])
print(f"Data loaded. Number of samples: {len(X)}")

# Train-test split
print("Splitting data into training and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

# Define the CNN model
print("Defining the CNN model...")
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(label_to_class), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print("Model compiled successfully.")

# Train the model
print("Training the model...")
early_stopping = EarlyStopping(patience=5, restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=20, validation_split=0.2, batch_size=32, verbose=2, callbacks=[early_stopping])

# Evaluate the model
print("Evaluating the model on the test set...")
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}, Test Loss: {test_loss:.4f}")

# Save the model and label mapping
print("Saving the model and label mapping...")
model.save("skin_disease_cnn_model.h5")
joblib.dump(label_to_class, "label_mapping.joblib")
print("Model and label mapping saved.")
