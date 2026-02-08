import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from src.data_utils import prepare_dataset


def create_simple_augmentation():
    return keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
    ])


def build_model(img_size=(224, 224)):
    inputs = keras.Input(shape=(*img_size, 3))
    x = create_simple_augmentation()(inputs)
    x = layers.Rescaling(scale=2.0, offset=-1.0)(x)  # [0,1] -> [-1,1]
    
    base_model = MobileNetV2(
        input_shape=(*img_size, 3),
        include_top=False,
        weights="imagenet",
        pooling='avg'
    )
    base_model.trainable = False
    x = base_model(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(2, activation="softmax")(x)  # 2 classes
    
    return keras.Model(inputs, outputs)


def train():
    # Load data
    X_train, X_val, y_train, y_val = prepare_dataset(
        batch_size=32,
        img_size=(224, 224)
    )
    
    print(f"Data loaded: {X_train.shape[0]} training, {X_val.shape[0]} validation")
    
    # Build and compile - using 2-class classification
    model = build_model()
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="categorical_crossentropy",  # For 2 classes
        metrics=["accuracy"]
    )
    
    # Callbacks
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            "model_best.keras",
            save_best_only=True,
            monitor="val_accuracy",
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
    ]
    
    # Train
    print("\nStarting training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=20,
        callbacks=callbacks,
        verbose=1
    )
    
    # Final evaluation
    best_epoch = len(history.history['val_accuracy'])
    print(f"\nTraining stopped at epoch {best_epoch}")
    print(f"Final Training Accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
    
    # Save final model
    model.save("model_final.keras")
    print(f"✓ Best model saved: model_best.keras")
    print(f"✓ Final model saved: model_final.keras")
    
    return model, history


if __name__ == "__main__":
    model, history = train()