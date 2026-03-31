from flask import Flask, request
import tensorflow as tf
import numpy as np
from PIL import Image
import os

app = Flask(__name__)

# Load model (safe check)
model_path = "models/detection_model/brain_tumor_cnn.keras"

if os.path.exists(model_path):
    model = tf.keras.models.load_model(model_path)
else:
    model = None

@app.route('/', methods=['GET', 'POST'])
def index():
    result = "No prediction yet"

    if request.method == 'POST':
        file = request.files['file']
        img = Image.open(file).resize((224, 224))
        img = np.array(img) / 255.0
        img = np.expand_dims(img, axis=0)

        if model:
            prediction = model.predict(img)
            result = str(np.argmax(prediction))
        else:
            result = "Model not found"

    return f"""
    <h1>🧠 Brain Tumor Detection</h1>
    <form method="POST" enctype="multipart/form-data">
        <input type="file" name="file" required>
        <button type="submit">Predict</button>
    </form>
    <h2>Result: {result}</h2>
    """

if __name__ == '__main__':
    app.run(debug=True)