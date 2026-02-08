import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image


def load_model(path="model_best.keras"):
    """Load trained model"""
    if not os.path.exists(path):
        # Try alternative names
        alternatives = ["model_best.keras", "model_final.keras", 
                       "model_best.h5", "model_final.h5"]
        for alt in alternatives:
            if os.path.exists(alt):
                path = alt
                break
        else:
            raise FileNotFoundError("No trained model found. Run train.py first.")
    
    print(f"Loading model: {path}")
    return keras.models.load_model(path)


def predict(img_path, model):
    """Predict cat or dog"""
    # Load and preprocess image
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img) / 255.0  # Scale to [0,1]
    x = x * 2.0 - 1.0  # Convert to [-1,1] for MobileNetV2
    x = np.expand_dims(x, axis=0)
    
    # Get prediction
    preds = model.predict(x, verbose=0)[0]
    
    # Model outputs 2 probabilities: [cat_prob, dog_prob]
    cat_prob, dog_prob = preds[0], preds[1]
    
    if dog_prob > cat_prob:
        return "Dog", dog_prob
    else:
        return "Cat", cat_prob


if __name__ == "__main__":
    # Load model
    model = load_model()
    
    # Test image
    test_images = [f for f in ["test.jpg", "test_cat.jpg", "test_dog.jpg"] 
                   if os.path.exists(f)]
    
    if test_images:
        for img_path in test_images:
            label, confidence = predict(img_path, model)
            print(f"{img_path}: {label} ({confidence:.2%} confidence)")
    else:
        print("No test images found.")
        print("Usage: python predict.py <image_path>")
        print("Or create test.jpg in current directory")