#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 16:15:55 2024

@author: cristofer
"""

import pandas as pd
import tkinter.filedialog
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from minisom import MiniSom
from pylab import bone, pcolor, colorbar, show

# Prompt user to select the file
filename = tkinter.filedialog.askopenfilename(title="Escolha o Arquivo", filetypes=[("Arquivos CSV", "*.csv"), ("Arquivos Excel", "*.xlsx"), ("Todos Arquivos", "*.*")])
if not filename: 
    exit(0)

# Load the dataset
data = pd.read_csv(filename, sep='\t')

# Preparing the data for the SOM (excluding the 'Item' column)
X = data[['Proteina', 'Carboidrato', 'gordura']].values
item_names = data['Item'].values  # Extract the item names for labeling

# Normalize the data using MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
X = scaler.fit_transform(X)

# Initialize the SOM
som = MiniSom(x=10, y=10, input_len=X.shape[1], sigma=1.0, learning_rate=0.5)
som.random_weights_init(X)

# Train the SOM
som.train_random(data=X, num_iteration=200)

# Visualizing the results
bone()  # Initializes the plot background
pcolor(som.distance_map().T)  # Plots the distance map of the SOM
colorbar()  # Adds a color legend

# Plot the winning nodes and display the item names
for i, x in enumerate(X):
    w = som.winner(x)  # Find the winning neuron
    plt.text(w[0] + 0.5, w[1] + 0.5, item_names[i], 
             fontsize=9, ha='center', va='center', color='white', weight='bold')

plt.title('SOM Visualization with Item Names')
plt.xlim([0, 10])
plt.ylim([0, 10])
plt.grid(True)

show()  # Show the plot
