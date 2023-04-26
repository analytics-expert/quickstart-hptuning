from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

def load_data_sklearn():
    """
    Loads, split and returns the digits dataset from sklearn.

    Returns:
    -------
    X_train, X_val: array-like
        The input samples for train and validation sets.
    y_train, y_val: array-like
        The target values for train and validation sets.
    """
    digits_dataset = load_digits()
    return train_test_split(digits_dataset.data, digits_dataset.target, 
                            test_size=0.2, random_state=42)


def load_data_tensorflow():
    """
    Load and preprocess  and returns the digits dataset from tensorflow.

    Returns:
        x_train: numpy array, training images
        x_test: numpy array, testing images
        y_train: numpy array, training labels
        y_test: numpy array, testing labels
    """
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    return x_train, x_test, y_train, y_test
