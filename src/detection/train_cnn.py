import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

# =========================
# PATH CONFIGURATION
# =========================
BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models", "detection_model")
MODEL_PATH = os.path.join(MODEL_DIR, "brain_tumor_cnn.keras")

os.makedirs(MODEL_DIR, exist_ok=True)

# =========================
# IMAGE PARAMETERS
# =========================
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 10
NUM_CLASSES = 4

# =========================
# DATA GENERATORS
# =========================
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

val_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)

# =========================
# MODEL ARCHITECTURE
# =========================
model = Sequential([
    Input(shape=(224, 224, 1)),

    Conv2D(32, (3, 3), activation="relu"),
    MaxPooling2D(),

    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D(),

    Conv2D(128, (3, 3), activation="relu"),
    MaxPooling2D(),

    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(NUM_CLASSES, activation="softmax")
])

# =========================
# COMPILE MODEL
# =========================
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# =========================
# TRAIN MODEL
# =========================
model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS
)

# =========================
# SAVE MODEL (IMPORTANT)
# =========================
model.save(MODEL_PATH)

print("\n✅ MODEL TRAINING COMPLETE")
print(f"✅ MODEL SAVED AT: {MODEL_PATH}")