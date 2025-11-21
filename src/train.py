# src/train.py
# src/train.py
import os
import tensorflow as tf
from tensorflow import keras

from src.model import CNN_model
from src.data_utils import prepare_dataset

def data_augmentation():
    return keras.Sequential([
        keras.layers.RandomFlip("horizontal"),
        keras.layers.RandomRotation(0.1),
        keras.layers.RandomZoom(0.1),
    ])

def train(
    batch_size=32,
    img_size=(150, 150),
    epochs=10,
    model_save="models/best_model.h5"
):
    # Load CIFAR-10 cats & dogs as numpy arrays
    X_train, X_val, y_train, y_val = prepare_dataset(
        batch_size=batch_size,
        img_size=img_size
    )

    num_classes = y_train.shape[1]

    # Build model using Functional API: Input -> Augment -> CNN
    inputs = keras.Input(shape=(*img_size, 3))
    x = data_augmentation()(inputs)
    cnn = CNN_model(input_shape=(*img_size, 3), n_classes=num_classes)
    outputs = cnn(x)
    model = keras.Model(inputs, outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    os.makedirs(os.path.dirname(model_save) or ".", exist_ok=True)

    callbacks = [
        keras.callbacks.ModelCheckpoint(model_save, save_best_only=True, monitor="val_loss", verbose=1),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1),
    ]

    model.summary()

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks
    )

    final_path = model_save.replace(".h5", "_final.h5")
    model.save(final_path)
    print(f"Best model: {model_save}")
    print(f"Final model: {final_path}")

if __name__ == "__main__":
    train()

