import os
import cv2
import numpy as np
import pandas as pd

def load_image(file_path):
    """Load an image, resize it to 128x128, and normalize it."""
    img = cv2.imread(file_path)
    img = cv2.resize(img, (128, 128))
    return img / 255.0

def load_data(metadata_path, image_paths):
    """Load images and labels from the metadata and image folders."""
    metadata = pd.read_csv(metadata_path)
    disease_labels = metadata['dx'].unique()
    class_to_label = {name: idx for idx, name in enumerate(disease_labels)}
    label_to_class = {v: k for k, v in class_to_label.items()}

    X, y = [], []
    for _, row in metadata.iterrows():
        image_id = row['image_id']
        label = class_to_label[row['dx']]
        
        # Find the image in both directories
        for image_path in image_paths:
            img_path = os.path.join(image_path, f"{image_id}.jpg")
            if os.path.exists(img_path):
                img = load_image(img_path)
                X.append(img)
                y.append(label)
                break
    
    return np.array(X), np.array(y), label_to_class
