import cv2
import pandas as pd
import os
import numpy as np
from mtcnn import MTCNN
import time

# Paths
train_image_folder = "../Data/trainset/trainset"
train_csv_path = "../Data/clean_dataset.csv"
output_comparison_file = "detection_comparison.csv"

# Load training data
train_data = pd.read_csv(train_csv_path)

# Initialize face detectors
mtcnn_detector = MTCNN()
ssd_detector = cv2.dnn.readNetFromCaffe(
    "deploy.prototxt",
    "res10_300x300_ssd_iter_140000.caffemodel"
)

# Function to preprocess images for SSD and MTCNN
def preprocess_image_for_ssd(image):
    """Resize image to 300x300 for SSD processing."""
    return cv2.resize(image, (300, 300), interpolation=cv2.INTER_AREA)

def preprocess_image_for_mtcnn(image, max_dim=2000):
    """
    Resize image to ensure maximum dimensions do not exceed max_dim.
    Retains aspect ratio.
    """
    h, w = image.shape[:2]
    if max(h, w) > max_dim:
        scaling_factor = max_dim / max(h, w)
        new_w = int(w * scaling_factor)
        new_h = int(h * scaling_factor)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return image

# Function to detect faces using SSD
def detect_faces_ssd(image):
    """Detect faces using SSD model."""
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))
    ssd_detector.setInput(blob)
    detections = ssd_detector.forward()

    face_locations = []
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.42:  # Confidence threshold
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            face_locations.append((startY, endX, endY, startX))  # (top, right, bottom, left)
    return face_locations

# Function to detect faces using MTCNN
def detect_faces_mtcnn(image):
    """Detect faces using MTCNN."""
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    detections = mtcnn_detector.detect_faces(rgb_image)
    face_locations = []
    for detection in detections:
        x, y, width, height = detection['box']
        face_locations.append((y, x + width, y + height, x))  # (top, right, bottom, left)
    return face_locations

# Process images and compare results
results = []
for idx, row in train_data.iterrows():
    # Convert image label to zero-padded filename
    image_name = str(row["Image name"]).zfill(4)  # Convert "33" -> "0033"
    jpg_path = os.path.join(train_image_folder, f"{image_name}.jpg")
    jpeg_path = os.path.join(train_image_folder, f"{image_name}.jpeg")

    # Check which file exists
    if os.path.exists(jpg_path):
        image_path = jpg_path
    elif os.path.exists(jpeg_path):
        image_path = jpeg_path
    else:
        print(f"File not found: {jpg_path} or {jpeg_path}")
        continue

    print(f"Processing image {idx + 1}/{len(train_data)}: {image_name}")

    try:
        original_image = cv2.imread(image_path)

        # Preprocess images for each model
        ssd_image = preprocess_image_for_ssd(original_image)
        mtcnn_image = preprocess_image_for_mtcnn(original_image)

        # Ground truth
        expected_faces = row["amount"]

        # SSD detection
        start_time = time.time()
        ssd_faces = detect_faces_ssd(ssd_image)
        ssd_time = time.time() - start_time

        # MTCNN detection
        start_time = time.time()
        mtcnn_faces = detect_faces_mtcnn(mtcnn_image)
        mtcnn_time = time.time() - start_time

        # Record results
        results.append({
            "Image Name": image_name,
            "Expected Faces": expected_faces,
            "SSD Detected Faces": len(ssd_faces),
            "MTCNN Detected Faces": len(mtcnn_faces),
            "SSD Processing Time (s)": ssd_time,
            "MTCNN Processing Time (s)": mtcnn_time
        })

    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        results.append({
            "Image Name": image_name,
            "Expected Faces": expected_faces,
            "SSD Detected Faces": "error",
            "MTCNN Detected Faces": "error",
            "SSD Processing Time (s)": "error",
            "MTCNN Processing Time (s)": "error"
        })

# Save comparison results to CSV
results_df = pd.DataFrame(results)
results_df.to_csv(output_comparison_file, index=False)

print(f"Detection comparison results saved to {output_comparison_file}")
