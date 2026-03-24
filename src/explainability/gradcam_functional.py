import tensorflow as tf
import numpy as np
import cv2

# =========================
# CONFIGURATION
# =========================
IMG_SIZE = 224
MODEL_WEIGHTS = "models/detection_model/brain_tumor_cnn.h5"
LAST_CONV_LAYER = "conv_last"

CLASS_NAMES = ["glioma", "meningioma", "no_tumor", "pituitary"]
TEST_IMAGE_PATH = "data/raw/glioma/Te-gl_1.jpg"


# =========================
# BUILD MODEL (FUNCTIONAL API)
# =========================
def build_model():
    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 1))

    x = tf.keras.layers.Conv2D(32, 3, activation="relu")(inputs)
    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.Conv2D(64, 3, activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.Conv2D(
        128, 3, activation="relu", name=LAST_CONV_LAYER
    )(x)
    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    outputs = tf.keras.layers.Dense(4, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    model.load_weights(MODEL_WEIGHTS)
    model.trainable = False

    return model


model = build_model()


# =========================
# LOAD & PREPROCESS IMAGE
# =========================
def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("❌ Image not found. Check path.")

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = img[..., np.newaxis]
    img = np.expand_dims(img, axis=0)
    return img


# =========================
# GRAD-CAM COMPUTATION
# =========================
def compute_gradcam(image, class_index):
    grad_model = tf.keras.Model(
        model.inputs,
        [model.get_layer(LAST_CONV_LAYER).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image)
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) if np.max(heatmap) != 0 else 1

    return heatmap


# =========================
# OVERLAY HEATMAP
# =========================
def overlay_heatmap(img_path, heatmap, alpha=0.4):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    heatmap = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)
    return overlay


# =========================
# MAIN
# =========================
def main():
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
    main()