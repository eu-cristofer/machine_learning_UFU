#!/usr/bin/env python
# coding: utf-8

# Machine Learning Project: MNIST Digit Classification with Various Regularization Techniques
# Aluno: Cristofer Antoni Souza Costa
# This script uses regularization techniques, data augmentation, and early stopping for enhanced neural network training.

# Import necessary libraries
import numpy as np  
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Input, Dropout
from keras import regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping

# Load the MNIST dataset from Keras
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Verify dataset shape to ensure it matches expectations
assert x_train.shape == (60000, 28, 28)  # Training images (60,000 samples, 28x28 pixels)
assert x_test.shape == (10000, 28, 28)   # Test images (10,000 samples, 28x28 pixels)
assert y_train.shape == (60000,)         # Training labels (60,000 labels)
assert y_test.shape == (10000,)          # Test labels (10,000 labels)

# Pick a sample image to visualize
sample = 1
image = x_train[sample]

# Plot the sample image with its corresponding label
plt.imshow(image)
plt.title(f"Label: {y_train[sample]}")
plt.show()

# Preprocess the data
x_train = x_train.astype('float32') / 255  # Normalize pixel values to [0, 1] range
x_train = np.expand_dims(x_train, -1)      # Reshape to (60000, 28, 28, 1) for compatibility with ImageDataGenerator

# Convert labels to categorical format (one-hot encoding)
train_labels = keras.utils.to_categorical(y_train).astype('float32')

# Split data into training and validation sets to monitor validation performance
x_train, x_valid, y_train, y_valid = train_test_split(x_train, train_labels, test_size=0.3, random_state=42)

# Define model architecture parameters
input_num_units = 784   # Number of input features (pixels in each image)
hidden1_num_units = 500 # Units in the first hidden layer
hidden2_num_units = 500 # Units in the second hidden layer
hidden3_num_units = 500 # Units in the third hidden layer
hidden4_num_units = 500 # Units in the fourth hidden layer
hidden5_num_units = 500 # Units in the fifth hidden layer
output_num_units = 10   # Output units (one per digit 0-9)

epochs = 10         # Number of training epochs
batch_size = 128    # Number of samples per gradient update

# Initialize a basic neural network model
model = Sequential([
    Input(shape=(input_num_units,)),       # Specify input shape
    Dense(hidden1_num_units, activation='relu'),  # First hidden layer with ReLU activation
    Dense(hidden2_num_units, activation='relu'),  # Second hidden layer with ReLU activation
    Dense(hidden3_num_units, activation='relu'),  # Third hidden layer with ReLU activation
    Dense(hidden4_num_units, activation='relu'),  # Fourth hidden layer with ReLU activation
    Dense(hidden5_num_units, activation='relu'),  # Fifth hidden layer with ReLU activation
    Dense(output_num_units, activation='softmax') # Output layer with softmax for multi-class classification
])

# Compile the model with categorical crossentropy loss and Adam optimizer
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model, tracking validation accuracy during training
trained_model_5d = model.fit(x_train.reshape(-1, 784), y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_valid.reshape(-1, 784), y_valid))

# ---- Applying L2 Regularization ----

model = Sequential([
    Input(shape=(input_num_units,)),
    Dense(hidden1_num_units, activation='relu', kernel_regularizer=regularizers.l2(0.0001)),
    Dense(hidden2_num_units, activation='relu', kernel_regularizer=regularizers.l2(0.0001)),
    Dense(hidden3_num_units, activation='relu', kernel_regularizer=regularizers.l2(0.0001)),
    Dense(hidden4_num_units, activation='relu', kernel_regularizer=regularizers.l2(0.0001)),
    Dense(hidden5_num_units, activation='relu', kernel_regularizer=regularizers.l2(0.0001)),
    Dense(output_num_units, activation='softmax')
])

# Compile and train with L2 regularization
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
trained_model_5d = model.fit(x_train.reshape(-1, 784), y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_valid.reshape(-1, 784), y_valid))

# ---- Applying L1 Regularization ----

model = Sequential([
    Input(shape=(input_num_units,)),
    Dense(hidden1_num_units, activation='relu', kernel_regularizer=regularizers.l1(0.0001)),
    Dense(hidden2_num_units, activation='relu', kernel_regularizer=regularizers.l1(0.0001)),
    Dense(hidden3_num_units, activation='relu', kernel_regularizer=regularizers.l1(0.0001)),
    Dense(hidden4_num_units, activation='relu', kernel_regularizer=regularizers.l1(0.0001)),
    Dense(hidden5_num_units, activation='relu', kernel_regularizer=regularizers.l1(0.0001)),
    Dense(output_num_units, activation='softmax')
])

# Compile and train with L1 regularization
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
trained_model_5d = model.fit(x_train.reshape(-1, 784), y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_valid.reshape(-1, 784), y_valid))

# ---- Applying Dropout Regularization ----

model = Sequential([
    Input(shape=(input_num_units,)),
    Dense(hidden1_num_units, activation='relu'),
    Dropout(0.25),
    Dense(hidden2_num_units, activation='relu'),
    Dropout(0.25),
    Dense(hidden3_num_units, activation='relu'),
    Dropout(0.25),
    Dense(hidden4_num_units, activation='relu'),
    Dropout(0.25),
    Dense(hidden5_num_units, activation='relu'),
    Dropout(0.25),
    Dense(output_num_units, activation='softmax')
])

# Compile and train with Dropout regularization
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
trained_model_5d = model.fit(x_train.reshape(-1, 784), y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_valid.reshape(-1, 784), y_valid))

# ---- Data Augmentation with ImageDataGenerator ----

# Reshape x_train to (samples, height, width, channels) for the generator
x_train = x_train.reshape(-1, 28, 28, 1)

# Initialize an ImageDataGenerator with ZCA whitening
datagen = ImageDataGenerator(zca_whitening=True, featurewise_center=True)

# Fit the generator on the training data (rank-4 input required)
datagen.fit(x_train)

# Train with augmented data using dropout model
for x_batch, y_batch in datagen.flow(x_train, y_train, batch_size=32):
    x_batch_flattened = x_batch.reshape(x_batch.shape[0], -1)
    break  # For simplicity, just one batch as demonstration

# ---- Using Early Stopping ----

# Define early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_accuracy', patience=2)

# Train the model with early stopping callback
trained_model_5d = model.fit(
    x_train.reshape(-1, 784),
    y_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(x_valid.reshape(-1, 784), y_valid),
    callbacks=[early_stopping]
)

