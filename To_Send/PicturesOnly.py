import os
import cv2
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, hamming_loss, recall_score
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
from tqdm import tqdm  # For progress bars
import tensorflow as tf
import joblib

# Paths
train_image_folder = "trainset"
test_image_folder = "testset"
train_csv_path = "clean_dataset.csv"
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# Helper Functions
def preprocess_image(image, target_size=(224, 224)):
    """Resize and normalize an image."""
    if image is None:
        return None
    image = cv2.resize(image, target_size)
    image = image / 255.0
    return image

def find_image_files(folder):
    """Find all image files in the folder with supported extensions."""
    valid_extensions = {".jpg", ".jpeg", ".png"}
    return [
        os.path.join(folder, f) for f in os.listdir(folder) 
        if os.path.splitext(f)[1].lower() in valid_extensions
    ]

def load_image(file_path):
    """Load an image file."""
    image = cv2.imread(file_path)
    return image

def load_train_data(image_folder, csv_path, label_encoder):
    """Load all training images and labels."""
    images, labels = [], []
    data = pd.read_csv(csv_path)
    for _, row in tqdm(data.iterrows(), total=len(data), desc="Loading training data"):
        image_name = str(row['image']).zfill(4)
        img_path = next(
            (os.path.join(image_folder, f) for f in os.listdir(image_folder) 
             if image_name in f and os.path.splitext(f)[1].lower() in {".jpg", ".jpeg", ".png"}), 
            None
        )

        if not img_path:
            print(f"Image {image_name} not found.")
            continue

        image = load_image(img_path)
        image = preprocess_image(image)
        if image is None:
            print(f"Failed to load or preprocess image {img_path}.")
            continue

        images.append(image)
        labels.append(row['label_name'])

    # Encode labels
    labels_encoded = label_encoder.fit_transform(labels)
    return np.array(images), np.array(labels_encoded)

def load_test_data(image_folder):
    """Load all test images dynamically."""
    images, filenames = [], []
    for img_path in tqdm(find_image_files(image_folder), desc="Loading test data"):
        filename = os.path.splitext(os.path.basename(img_path))[0]
        image = load_image(img_path)
        image = preprocess_image(image)
        if image is None:
            print(f"Failed to load or preprocess image {img_path}.")
            continue
        images.append(image)
        filenames.append(filename)
    return np.array(images), filenames

# Load and preprocess data
print("Loading training data...")
label_encoder = LabelEncoder()
X_train, y_train = load_train_data(train_image_folder, train_csv_path, label_encoder)

print("Loading test data...")
X_test, test_filenames = load_test_data(test_image_folder)

# Define models
models = {
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(kernel="rbf", probability=True),
    "KNN": KNeighborsClassifier(n_neighbors=3, metric="euclidean"),
    "VGGFace2": VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3)),
    "ResNet50": ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3)),
    "MobileNet": MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3)),
    #"XGBoost": XGBClassifier(eval_metric="logloss"),
}

# Train and evaluate models
results = []
best_model = None
best_model_name = None
lowest_hamming_loss = float("inf")

# Paths
log_filename = os.path.join(output_dir, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

# Open log file for writing
with open(log_filename, "w") as log_file:
    log_file.write("Training Log\n")
    log_file.write("=" * 40 + "\n")

    for model_name, model in tqdm(models.items(), desc="Training Models"):
        print(f"Training {model_name}...")
        log_file.write(f"Training {model_name}...\n")

        if model_name in {"VGGFace2", "ResNet50", "MobileNet"}:
            # Pretrained models (Keras)
            x = Flatten()(model.output)
            x = Dense(128, activation="relu")(x)
            output = Dense(len(label_encoder.classes_), activation="softmax")(x)
            model = Model(inputs=model.input, outputs=output)

            model.compile(optimizer=Adam(learning_rate=0.0001), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
            model.fit(X_train, y_train, epochs=5, batch_size=16, verbose=1)

            train_predictions = np.argmax(model.predict(X_train), axis=1)
            test_predictions = np.argmax(model.predict(X_test), axis=1)
        else:
            # Non-pretrained models (Scikit-learn) with progress bar
            train_flattened = X_train.reshape(len(X_train), -1)
            test_flattened = X_test.reshape(len(X_test), -1)

            print(f"Training {model_name} with progress bar...")
            for _ in tqdm(range(1), desc=f"Fitting {model_name}"):
                model.fit(train_flattened, y_train)

            train_predictions = model.predict(train_flattened)
            test_predictions = model.predict(test_flattened)

        # Evaluate model
        accuracy = accuracy_score(y_train, train_predictions)
        hamming = hamming_loss(y_train, train_predictions)
        recall = recall_score(y_train, train_predictions, average="macro")

        # Log training results
        results.append({
            "model": model_name,
            "accuracy": accuracy,
            "hamming_loss": hamming,
            "recall": recall,
        })
        log_file.write(f"Model: {model_name}, Accuracy: {accuracy:.4f}, Hamming Loss: {hamming:.4f}, Recall: {recall:.4f}\n")
        print(f"Model: {model_name}, Accuracy: {accuracy:.4f}, Hamming Loss: {hamming:.4f}, Recall: {recall:.4f}")

        # Save model during training
        current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
        if model_name in {"VGGFace2", "ResNet50", "MobileNet"}:
            save_model(model, os.path.join(output_dir, f"{model_name}_{accuracy:.4f}_{current_datetime}.h5"))
        else:
            joblib.dump(model, os.path.join(output_dir, f"{model_name}_{accuracy:.4f}_{current_datetime}.pkl"))

        # Check for the best model
        if hamming < lowest_hamming_loss:
            best_model = model
            best_model_name = model_name
            lowest_hamming_loss = hamming

        # Decode test predictions
        test_labels = label_encoder.inverse_transform(test_predictions)

        # Save submission
        submission_filename = f"{model_name}_{accuracy:.4f}_{current_datetime}.csv"
        submission_data = pd.DataFrame({
            'image': test_filenames,
            'label_name': test_labels
        })
        submission_data.to_csv(os.path.join(output_dir, submission_filename), index=False)
        print(f"Submission file saved as {submission_filename}")

    # Log the best model
    log_file.write(f"\nBest Model: {best_model_name} with Hamming Loss: {lowest_hamming_loss:.4f}\n")
    print(f"Best model details saved to '{log_filename}'.")

# Save comparison results
results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(output_dir, "model_comparison_results.csv"), index=False)
print("Results saved to 'model_comparison_results.csv'.")

