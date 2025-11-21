# src/data_utils.py
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

def prepare_dataset(
    batch_size=32,          # kept for API compatibility, not used directly
    img_size=(150, 150),
    val_split=0.2,
    seed=42
):
    """
    Use built-in CIFAR-10 dataset and keep only:
      - class 3 = cat
      - class 5 = dog

    Returns:
        X_train, X_val, y_train, y_val
        where X are images (float32, scaled 0–1),
        and y are one-hot labels with 2 classes: [cat, dog]
    """

    # 1) Load CIFAR-10 (inbuilt, very reliable)
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # 2) Combine train + test, then filter only cat (3) and dog (5)
    X = np.concatenate([x_train, x_test], axis=0)      # uint8, shape (N, 32, 32, 3)
    y = np.concatenate([y_train, y_test], axis=0).flatten()  # shape (N,)

    # mask for cat (3) and dog (5)
    mask = np.isin(y, [3, 5])
    X = X[mask]
    y = y[mask]

    # Map labels: 3 -> 0 (cat), 5 -> 1 (dog)
    y = np.where(y == 3, 0, 1)

    print("Total cat/dog images:", X.shape[0])
    print("Cats:", np.sum(y == 0), "Dogs:", np.sum(y == 1))

    # 3) Resize to desired img_size (150x150) and scale to [0,1]
    X = tf.image.resize(X, img_size)   # returns Tensor
    X = X.numpy().astype("float32") / 255.0

    # 4) Train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=val_split,
        stratify=y,
        random_state=seed
    )

    # 5) One-hot encode labels: 2 classes
    num_classes = 2
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_val   = tf.keras.utils.to_categorical(y_val,   num_classes)

    print("Train size:", X_train.shape[0], "Val size:", X_val.shape[0])
    return X_train, X_val, y_train, y_val

