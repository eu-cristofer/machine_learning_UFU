# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 10:11:12 2024

@author: EMKA
"""

import numpy as np  # Import NumPy for numerical operations
import pandas as pd  # Import pandas for data manipulation
import tkinter.filedialog  # Import Tkinter to open file dialogs
# import matplotlib.pyplot as plt  # Import Matplotlib (although not used in this script)
# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay  # Commented out, used for confusion matrix display

"""
NOTE:
=====

    The folowing statements are intent to deal with Keras import error:
        2024-09-12 10:37:00.075072: I tensorflow/core/util/port.cc:153] oneDNN
        custom operations are on. You may see slightly different numerical results
        due to floating-point round-off errors from different computation orders.
        To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
"""
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from keras.models import Sequential  # Import Sequential model from Keras
from keras.layers import Dense  # Import Dense (fully connected) layers from Keras
from sklearn import preprocessing  # Import preprocessing tools from scikit-learn

filename = tkinter.filedialog.askopenfilename(
    title="Escolha o Arquivo",
    filetypes=[("Arquivos CSV", "*.csv"),
               ("Arquivos Excel", "*.xlsx"),
               ("Todos Arquivos", "*.*")]
)

# Step 2: Load the data if a file was selected, otherwise exit the program
if filename:
    df = pd.read_csv(filename)
    dados = np.array(df.iloc[:,1:].values)  # Excluding the first row
else:
    exit(0)  # Exit the program if no file was selected

# Step 3: Select specific columns as features (par) and target labels (tag)
par_col = [0, 1, 2, 3]                  # Indices of the columns used as features
par = dados[:, par_col]                 # Extract feature columns from the data
tag = dados[:, 4].astype(int)           # Extract the target labels (column 4) and convert them to integers

# Step 4: Binarize the target labels (tag) into a binary matrix
lb = preprocessing.LabelBinarizer()     # Initialize a LabelBinarizer to convert labels into binary form
lb.fit([3, 4, 5, 6, 7, 8, 9])           # Fit the LabelBinarizer with the possible label values (3 to 9)
tag = lb.transform(tag)                 # Transform the original target labels into a binary matrix

# Step 5: Normalize the feature data (par) by subtracting the mean and dividing by the standard deviation
n_par = len(par_col)  # Number of features (columns in par)
for i in range(n_par):  # Loop over each feature
    med = np.mean(par[:, i])  # Calculate the mean of the feature
    st = np.std(par[:, i])  # Calculate the standard deviation of the feature
    aux = (par[:, i] - med) / st  # Normalize the feature (Z-score normalization)
    aux[aux > 3] = 3  # Clip values above 3 standard deviations
    aux[aux < -3] = -3  # Clip values below -3 standard deviations
    par[:, i] = aux  # Update the normalized feature

# Step 6: Construct a Keras Sequential neural network model
model = Sequential()  # Initialize a Sequential model
model.add(Dense(45, input_shape=(n_par,), activation='relu'))  # Add the first Dense layer with 45 neurons, ReLU activation
model.add(Dense(14, activation='relu'))  # Add the second Dense layer with 14 neurons, ReLU activation
model.add(Dense(7, activation='sigmoid'))  # Add the output Dense layer with 7 neurons, sigmoid activation (binary classification)

# Step 7: Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])  # Compile the model with binary crossentropy loss and Adam optimizer

# Step 8: Train the model using the training data (par) and labels (tag)
model.fit(par, tag, epochs=15, batch_size=10)  # Train the model for 15 epochs, with a batch size of 10

# Step 9: Evaluate the model on the training data
_, accuracy = model.evaluate(par, tag)  # Evaluate the model and get the accuracy
print('Accuracy: %.2f' % (accuracy * 100))  # Print the accuracy as a percentage
