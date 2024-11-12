import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

# To stop potential randomness
np.random.seed(128)

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Checking the shapes
assert x_train.shape == (60000, 28, 28), "x_train shape should be (60000, 28, 28)"
assert x_test.shape == (10000, 28, 28), "x_test shape should be (10000, 28, 28)"
assert y_train.shape == (60000,), "y_train shape should be (60000,)"
assert y_test.shape == (10000,), "y_test shape should be (10000,)"

# Displaying an example image from the dataset
sample = 1  # Sample index to display
plt.imshow(
    x_train[sample],
    #cmap="gray"
)
plt.title(f"Sample Image - Label: {y_train[sample]}")
plt.show()
