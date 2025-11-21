# src/predict.py
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import numpy as np

def load_model(path=None):
    if path is None:
        path = os.path.join("models", "best_model.h5")
    return keras.models.load_model(path)

def predict(img_path, model, img_size=(150,150)):
    img = image.load_img(img_path, target_size=img_size)
    x = image.img_to_array(img) / 255.0
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)
    label = np.argmax(preds, axis=1)[0]
    return 'Dog' if label==1 else 'Cat'

if __name__ == "__main__":
    model = load_model()
    img_path = 'test.jpg'
    result = predict(img_path, model)
    print(f"Prediction: {result}")
