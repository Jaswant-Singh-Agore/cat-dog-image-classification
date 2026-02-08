import numpy as np
from flask import Flask, request, render_template
from tensorflow import keras
from PIL import Image
import os

MODEL_PATH = "models/model_final.keras"
IMG_SIZE = (150, 150)
CLASS_NAMES = ["Cat", "Dog"]

print(f"Loading model from: {MODEL_PATH}")
model = keras.models.load_model(MODEL_PATH)

app = Flask(__name__)

def preprocess_image(file):
    img = Image.open(file).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img).astype("float32") / 255.0
    return np.expand_dims(arr, axis=0)  

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    error = None

    if request.method == "POST":
        file = request.files.get("file")

        if not file:
            error = "Please upload an image."
        else:
            x = preprocess_image(file)
            preds = model.predict(x)[0]          
            label_idx = preds.argmax()           
            prediction = CLASS_NAMES[label_idx]  
            confidence = round(preds[label_idx] * 100, 2)

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        error=error
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
