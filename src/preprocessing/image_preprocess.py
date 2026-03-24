import os
import cv2
import numpy as np

# =========================
# CONFIGURATION
# =========================
RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"
IMG_SIZE = 224


# =========================
# IMAGE PREPROCESSING
# =========================
def preprocess_image(image_path):
    """
    Preprocess a single MRI image:
    1. Read image
    2. Convert to grayscale
    3. Resize to fixed size
    4. Normalize pixel values (0–1)
    """
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError("Image not readable")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = image / 255.0

    return image


# =========================
# DATASET PREPROCESSING
# =========================
def preprocess_dataset():

    # Ensure processed directory exists and is a FOLDER
    if os.path.exists(PROCESSED_DATA_DIR):
        if not os.path.isdir(PROCESSED_DATA_DIR):
            raise Exception(
                "'data/processed' exists but is NOT a folder. Delete it and try again."
            )
    else:
        os.makedirs(PROCESSED_DATA_DIR)

    # Loop through tumor classes
    for tumor_class in os.listdir(RAW_DATA_DIR):
        class_path = os.path.join(RAW_DATA_DIR, tumor_class)

        # Skip if not a folder
        if not os.path.isdir(class_path):
            continue

        save_class_path = os.path.join(PROCESSED_DATA_DIR, tumor_class)
        os.makedirs(save_class_path, exist_ok=True)

        print(f"Processing class: {tumor_class}")

        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)

            try:
                processed_img = preprocess_image(img_path)
                save_path = os.path.join(save_class_path, img_name)

                cv2.imwrite(
                    save_path,
                    (processed_img * 255).astype(np.uint8)
                )

            except Exception as e:
                print(f"Skipping {img_name}: {e}")

    print("\n✅ MRI preprocessing completed successfully.")


# =========================
# ENTRY POINT
# =========================
if __name__ == "__main__":
    print("Running from:", os.getcwd())
    preprocess_dataset()