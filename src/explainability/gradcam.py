import tensorflow as tf
import numpy as np
import cv2
import os

# =========================
# CONFIG
# =========================
MODEL_PATH = "models/detection_model/brain_tumor_cnn.h5"
IMG_SIZE = 224
LAST_CONV_LAYER = "conv2d_2"

CLASS_NAMES = ["glioma", "meningioma", "no_tumor", "pituitary"]
TEST_IMAGE_PATH = "data/raw/glioma/Te-gl_1.jpg"

# =========================
# LOAD MODEL (SAFE)
# =========================
model = tf.keras.models.load_model(MODEL_PATH)
model.trainable = False

# Force build
_ = model(tf.zeros((1, IMG_SIZE, IMG_SIZE, 1)))


# =========================
# LOAD IMAGE
# =========================
def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Image not found")

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)
    return img


# =========================
# GRAD-CAM (ROBUST)
# =========================
def compute_gradcam(image, class_index):
    conv_layer = model.get_layer(LAST_CONV_LAYER)

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[conv_layer.output, model.outputs[0]]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image)
        tape.watch(conv_outputs)
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)

    if grads is None:
        raise RuntimeError("Gradients are None. Check model & layer name.")

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) if np.max(heatmap) != 0 else 1

    return heatmap.numpy()


# =========================
# OVERLAY
# =========================
def overlay_heatmap(img_path, heatmap, alpha=0.4):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    heatmap = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    return cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)


# =========================
# MAIN
# =========================
def run_gradcam():
    image = load_image(TEST_IMAGE_PATH)

    preds = model.predict(image)
    class_index = int(np.argmax(preds))
    confidence = float(np.max(preds))
    class_name = CLASS_NAMES[class_index]

    heatmap = compute_gradcam(image, class_index)
    overlay = overlay_heatmap(TEST_IMAGE_PATH, heatmap)

    output_path = "gradcam_output.png"
    cv2.imwrite(output_path, overlay)

    print("\n✅ GRAD-CAM SUCCESS")
    print(f"Predicted Class : {class_name}")
    print(f"Confidence      : {confidence:.2f}")
    print(f"Saved File      : {output_path}")


if __name__ == "__main__":
    run_gradcam()