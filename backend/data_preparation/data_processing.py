# process.py
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

def preprocess_and_split_data(X, y, batch_size=50, num_classes=3):
    """
    Normalizes the image data, encodes the labels, and splits the dataset into training, validation, and test sets.

    Args:
    - X (ndarray): Array of images.
    - y (ndarray): Array of labels.
    - batch_size (int): Size of batches to process at a time for normalization.
    - num_classes (int): The number of classes in the labels.
    - test_size (float): The proportion of the dataset to be used as the test set.
    - random_state (int): Seed for reproducibility.

    Returns:
    - X_train (ndarray): Training data.
    - X_val (ndarray): Validation data.
    - X_test (ndarray): Test data.
    - y_train (ndarray): Training labels.
    - y_val (ndarray): Validation labels.
    - y_test (ndarray): Test labels.
    """
    # Normalize the images (convert to float32 and scale to [0, 1] range)
    for i in range(0, X.shape[0], batch_size):
        X[i:i+batch_size] = X[i:i+batch_size].astype('float32') / 255.0

    # One-hot encode the labels
    y = to_categorical(y, num_classes=num_classes)

    # Split the data into training, validation, and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test
