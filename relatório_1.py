# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 10:11:12 2024

@author: Cristofer Antoni Souza Costa
"""

import os
# Disable oneDNN optimizations to avoid floating-point round-off errors
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import pandas as pd
import tkinter.filedialog as filedialog
from keras.models import Sequential
from keras.layers import Dense
from sklearn import preprocessing


# Step 1: Open file dialog to select a CSV or Excel file
filename = filedialog.askopenfilename(
    title="Escolha o Arquivo",
    filetypes=[("Arquivos CSV", "*.csv"),
               ("Arquivos Excel", "*.xlsx"),
               ("Todos Arquivos", "*.*")]
)


# Step 2: Load the data if a file was selected, otherwise exit the program
# database: BASE.csv, Aula 5
if filename:
    df = pd.read_csv(filename)
    data = df.iloc[:, 1:].values  # Excluding the first column (assumed to be non-feature)
else:
    exit(0)

# Step 3: Select specific columns as features (par) and target labels (tag)
par_cols = [0, 1, 2, 3]             # Indices of columns used as features
par = data[:, par_cols]             # Extract features
tag = data[:, 4].astype(int)        # Extract and convert target labels to integers

# Step 4: Binarize target labels (tag) into a binary matrix
lb = preprocessing.LabelBinarizer()
lb.fit([3, 4, 5, 6, 7, 8, 9])       # Fit with possible label values from 3 to 9
tag = lb.transform(tag)

# Step 5: Normalize feature data (par) using Z-score normalization
for i in range(par.shape[1]):
    mean, std = np.mean(par[:, i]), np.std(par[:, i])
    par[:, i] = np.clip((par[:, i] - mean) / std, -3, 3)  # Normalize and clip between -3 and 3

# Step 6: Construct a Sequential neural network model using Keras
model = Sequential([
    Dense(45, input_shape=(par.shape[1],), activation='relu'),  # First layer with 45 neurons, ReLU activation
    Dense(14, activation='relu'),  # Second layer with 14 neurons, ReLU activation
    Dense(7, activation='sigmoid')  # Output layer with 7 neurons, sigmoid activation
])

# Step 7: Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Step 8: Train the model
model.fit(par,
          tag,
          epochs=15,
          batch_size=10,
          verbose=1)

# Step 9: Evaluate the model
loss, accuracy = model.evaluate(par, tag, verbose=0)
print(f'Accuracy: {accuracy * 100:.2f}%')
