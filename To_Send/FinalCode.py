import os
import cv2
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, hamming_loss, recall_score, f1_score
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from tensorflow.keras.applications import ResNet50, MobileNetV2, VGG16
from tensorflow.keras.models import Model, save_model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from mtcnn import MTCNN
from datetime import datetime
import tensorflow as tf
import joblib
from PIL import Image
import pillow_heif
import tqdm

# Paths
train_image_folder = "../trainset"
test_image_folder = "../testset"
train_csv_path = "clean_dataset.csv"
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

def preprocess_image(image, target_size=(224, 224)):
    """Resize and normalize an image."""
    if image is None:
        return None
    image = cv2.resize(image, target_size) 
    image = image / 255.0  
    return image

def find_image_files(folder):
    """Find all image files in the folder with supported extensions."""
    valid_extensions = {".jpg", ".jpeg", ".png", ".heic", ".mp4"}
    return [
        os.path.join(folder, f) for f in os.listdir(folder)
        if os.path.splitext(f)[1].lower() in valid_extensions
    ]

def load_file_for_opencv(file_path):
    """
    Load a file (HEIC, MP4, or JPG) and return a list of frames for OpenCV.
    """
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

def load_train_data(image_folder, csv_path, label_encoder):
    """Load all training images and labels."""
    images, labels = [], []
    data = pd.read_csv(csv_path)
    for _, row in tqdm.tqdm(data.iterrows(), total=len(data), desc="Loading training data"):
        image_name = str(row['image']).zfill(4)
        img_path = next(
            (os.path.join(image_folder, f) for f in os.listdir(image_folder)
             if image_name in f and os.path.splitext(f)[1].lower() in {".jpg", ".jpeg", ".png", ".heic", ".mp4"}),
            None
        )

        if not img_path:
            print(f"Image {image_name} not found.")
            continue

        frames = load_file_for_opencv(img_path)
        if not frames:
            print(f"Failed to load file: {img_path}.")
            continue

        frame = preprocess_image(frames[0])
        if frame.shape != (224, 224, 3):
            print(f"Unexpected shape for {img_path}. Skipping.")
            continue

        images.append(frame)
        labels.append(row['label_name'])

    labels_encoded = label_encoder.fit_transform(labels)
    return np.array(images), np.array(labels_encoded)

def load_test_data(image_folder):
    """Load all test images dynamically."""
    images, filenames = [], []
    for img_path in tqdm.tqdm(find_image_files(image_folder), desc="Loading test data"):
        filename = os.path.splitext(os.path.basename(img_path))[0]
        frames = load_file_for_opencv(img_path)
        if not frames:
            print(f"Failed to load file: {img_path}.")
            continue

        frame = preprocess_image(frames[0])
        images.append(frame)
        filenames.append(filename)
    return np.array(images), filenames

# Load and preprocess data
print("Loading training data...")
label_encoder = LabelEncoder()
X_train, y_train = load_train_data(train_image_folder, train_csv_path, label_encoder)

# Save the Label Encoder
label_encoder_file = os.path.join(output_dir, "label_encoder.pkl")
joblib.dump(label_encoder, label_encoder_file)
print(f"Label encoder saved to '{label_encoder_file}'.")

print("Loading test data...")
X_test, test_filenames = load_test_data(test_image_folder)

# Define models
models = {
    "MobileNet": MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3)),
}

# Train and evaluate models
results = []
best_model = None
best_model_name = None
lowest_hamming_loss = float("inf")

log_filename = os.path.join(output_dir, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

with open(log_filename, "w") as log_file:
    log_file.write("Training Log\n")
    log_file.write("=" * 40 + "\n")

    for model_name, model in tqdm.tqdm(models.items(), desc="Training Models"):
        print(f"Training {model_name}...")
        log_file.write(f"Training {model_name}...\n")

        x = Flatten()(model.output)
        x = Dense(128, activation="relu")(x)
        output = Dense(len(label_encoder.classes_), activation="softmax")(x)
        model = Model(inputs=model.input, outputs=output)

        model.compile(optimizer=Adam(learning_rate=0.0001), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        model.fit(X_train, y_train, epochs=5, batch_size=16, verbose=1)

        train_predictions = np.argmax(model.predict(X_train), axis=1)
        test_predictions = np.argmax(model.predict(X_test), axis=1)

        accuracy = accuracy_score(y_train, train_predictions)
        hamming = hamming_loss(y_train, train_predictions)
        recall = recall_score(y_train, train_predictions, average="macro")
        f1 = f1_score(y_train, train_predictions, average="macro")  

        results.append({
            "model": model_name,
            "accuracy": accuracy,
            "hamming_loss": hamming,
            "recall": recall,
            "f1_score": f1
        })
        log_file.write(f"Model: {model_name}, Accuracy: {accuracy:.4f}, Hamming Loss: {hamming:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}\n")
        print(f"Model: {model_name}, Accuracy: {accuracy:.4f}, Hamming Loss: {hamming:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

        # Save model during training
        current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_model(model, os.path.join(output_dir, f"{model_name}_{accuracy:.4f}_{current_datetime}.h5"))

        # Check for the best model
        if hamming < lowest_hamming_loss:
            best_model = model
            best_model_name = model_name
            lowest_hamming_loss = hamming

# Decode test labels
test_labels = label_encoder.inverse_transform(test_predictions)

# Save submission
submission_filename = "submission.csv"
submission_data = pd.DataFrame({
    'image': [f"image_{int(name):04d}" for name in test_filenames],
    'label_name': test_labels  # Directly use decoded test labels
})
submission_data.to_csv(os.path.join(output_dir, submission_filename), index=False)
print(f"Submission file saved as {submission_filename}")

# Save comparison results
results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(output_dir, "model_comparison_results.csv"), index=False)
print("Results saved to 'model_comparison_results.csv'.")
