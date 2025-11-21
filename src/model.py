# src/model.py
from tensorflow import keras
from tensorflow.keras import layers, models

def CNN_model(input_shape=(150,150,3), n_classes=2):
    model = models.Sequential([
        layers.Conv2D(32,(3,3), padding='same', input_shape=input_shape),
        layers.Activation('relu'),
        layers.BatchNormalization(),

        layers.Conv2D(32,(3,3), padding='same'),
        layers.Activation('relu'),
        layers.MaxPool2D((2,2)),
        layers.Dropout(0.25),

        layers.Conv2D(64,(3,3), padding='same'),
        layers.Activation('relu'),
        layers.BatchNormalization(),

        layers.Conv2D(64,(3,3), padding='same'),
        layers.Activation('relu'),
        layers.MaxPool2D((2,2)),
        layers.Dropout(0.25),

        layers.Conv2D(128,(3,3), padding='same'),
        layers.Activation('relu'),
        layers.BatchNormalization(),
        layers.MaxPool2D((2,2)),
        layers.Dropout(0.25),

        layers.Flatten(),
        layers.Dense(256),
        layers.Activation('relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.25),

        layers.Dense(n_classes, activation='softmax'),
    ])
    return model
