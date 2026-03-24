from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
import tensorflow as tf
from PIL import Image
import io

# =========================
# APP INIT
# =========================
app = FastAPI(title="NeuroTwin-AI API")

MODEL_PATH = "models/detection_model/brain_tumor_cnn.h5"
IMG_SIZE = 224
CLASS_NAMES = ["glioma", "meningioma", "no_tumor", "pituitary"]

model = tf.keras.models.load_model(MODEL_PATH)


# =========================
# IMAGE PREPROCESS
# =========================
def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("L")
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image) / 255.0
    image = image.reshape(1, IMG_SIZE, IMG_SIZE, 1)
    return image


# =========================
# ROUTES
# =========================
@app.get("/")
def home():
    return {"message": "NeuroTwin-AI API is running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = preprocess_image(image_bytes)

    predictions = model.predict(image)
    class_index = int(np.argmax(predictions))
    confidence = float(np.max(predictions))

    return {
        "prediction": CLASS_NAMES[class_index],
        "confidence": round(confidence, 4)
    }


# =========================
# RUN SERVER
# =========================
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)