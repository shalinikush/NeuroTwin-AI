import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

IMG_SIZE = 256
DATA_DIR = "data/segmentation"
BATCH_SIZE = 4
EPOCHS = 5


# =========================
# LOAD DATA
# =========================
def load_data():
    images, masks = [], []

    image_dir = os.path.join(DATA_DIR, "images")
    mask_dir = os.path.join(DATA_DIR, "masks")

    for img_name in os.listdir(image_dir):
        img_path = os.path.join(image_dir, img_name)
        mask_path = os.path.join(mask_dir, img_name)

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) / 255.0
        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE)) / 255.0

        images.append(img[..., np.newaxis])
        masks.append(mask[..., np.newaxis])

    return np.array(images), np.array(masks)


# =========================
# U-NET MODEL
# =========================
def build_unet():
    inputs = layers.Input((IMG_SIZE, IMG_SIZE, 1))

    c1 = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(32, 3, activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D()(c1)

    c2 = layers.Conv2D(64, 3, activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(64, 3, activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D()(c2)

    c3 = layers.Conv2D(128, 3, activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(128, 3, activation='relu', padding='same')(c3)

    u1 = layers.UpSampling2D()(c3)
    u1 = layers.concatenate([u1, c2])
    c4 = layers.Conv2D(64, 3, activation='relu', padding='same')(u1)

    u2 = layers.UpSampling2D()(c4)
    u2 = layers.concatenate([u2, c1])
    c5 = layers.Conv2D(32, 3, activation='relu', padding='same')(u2)

    outputs = layers.Conv2D(1, 1, activation='sigmoid')(c5)

    model = models.Model(inputs, outputs)
    return model


# =========================
# TRAIN
# =========================
images, masks = load_data()
model = build_unet()

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

model.fit(
    images, masks,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS
)

os.makedirs("models/segmentation_model", exist_ok=True)
model.save("models/segmentation_model/unet_model.keras")

print("\n✅ U-Net segmentation model trained & saved")