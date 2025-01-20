import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import cv2
from PIL import Image
import pillow_heif
import joblib

# Paths
test_image_folder = "testset"  
model_path = "MobileNet.h5"  
output_dir = "outputCanvas2"  
os.makedirs(output_dir, exist_ok=True)
train_csv_path = "clean_dataset.csv" 
data = pd.read_csv(train_csv_path)
label_encoder = LabelEncoder()
label_encoder.fit(data['label_name'])
joblib.dump(label_encoder, "label_encoder.pkl")

def preprocess_image(image, target_size=(224, 224)):
    
    if image is None:
        return None
    image = cv2.resize(image, target_size) 
    image = image / 255.0  
    return image

def find_image_files(folder):
    valid_extensions = {".jpg", ".jpeg", ".png", ".heic", ".mp4"}
    return [
        os.path.join(folder, f) for f in os.listdir(folder)
        if os.path.splitext(f)[1].lower() in valid_extensions
    ]

def load_file_for_opencv(file_path):
    file_ext = os.path.splitext(file_path)[1].lower()
    frames = []

    if file_ext in [".heic", ".jpg", ".jpeg", ".png"]:
        try:
            if file_ext == ".heic":
                heif_file = pillow_heif.read_heif(file_path)
                image = Image.frombytes(
                    heif_file.mode, heif_file.size, heif_file.data, "raw"
                )
                image_array = np.array(image)
                image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
                frames.append(image_array)
            else:
                image_array = cv2.imread(file_path)
                if image_array is None:
                    raise ValueError("Failed to load image.")
                frames.append(image_array)
        except Exception as e:
            print(f"Error loading image file {file_path}: {e}")

    elif file_ext in [".mp4", ".avi", ".mkv"]:
        try:
            video_capture = cv2.VideoCapture(file_path)
            while video_capture.isOpened():
                ret, frame = video_capture.read()
                if not ret:
                    break
                frames.append(frame)
            video_capture.release()
        except Exception as e:
            print(f"Error loading video file {file_path}: {e}")

    else:
        print(f"Unsupported file format: {file_ext}")

    return frames

def load_test_data(image_folder):
    images, filenames = [], []
    for img_path in tqdm(find_image_files(image_folder), desc="Loading test data"):
        filename = os.path.splitext(os.path.basename(img_path))[0]
        frames = load_file_for_opencv(img_path)
        if not frames:
            print(f"Failed to load file: {img_path}.")
            continue

        frame = preprocess_image(frames[0])
        if frame is None:
            print(f"Skipping {img_path} due to preprocessing failure.")
            continue

        images.append(frame)
        filenames.append(filename)
    return np.array(images), filenames

print("Loading pre-trained MobileNet model...")
model = load_model(model_path)

print("Loading test data...")
X_test, test_filenames = load_test_data(test_image_folder)

print("Running predictions...")
test_predictions = np.argmax(model.predict(X_test), axis=1)

test_labels = label_encoder.inverse_transform(test_predictions)

submission_filename = os.path.join(output_dir, "submission.csv")
submission_data = pd.DataFrame({
    'image': [f"image_{int(name):04d}" for name in test_filenames],
    'label_name': test_labels  # Use the decoded labels
})
submission_data.to_csv(submission_filename, index=False)
print(f"Submission file saved as {submission_filename}")

print("Test evaluation and submission generation complete.")
