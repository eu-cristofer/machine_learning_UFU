#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 11:05:32 2024

@author: cristofer
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import tensorflow as tf
import keras

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Verify the shapes of the loaded data
assert x_train.shape == (60000, 28, 28)
assert x_test.shape == (10000, 28, 28)
assert y_train.shape == (60000,)
assert y_test.shape == (10000,)

# Plot a sample image
sample = 1
image = x_train[sample]

# Plot the sample
plt.imshow(image)
plt.show()

# Data preprocessing
x_train = x_train.astype('float32')
x_train /= 255
x_train = x_train.reshape(-1, 784).astype('float32')
x_train = x_train[:20000, :]  # Use a subset of the data
train_labels = keras.utils.to_categorical(y_train).astype('float32')
train_labels = train_labels[:20000, :]  # Use a subset of labels