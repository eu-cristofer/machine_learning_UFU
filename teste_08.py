import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import time

# 1. Data Preprocessing
# Load the dataset
data = pd.read_csv('data_files\\teste.csv')  # Ensure 'teste.csv' is in the same directory or provide the full path

# Separate features and target
X = data.drop(columns=['target'])
y = data['target']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# 2. Define the Neural Network Model and Objective Function
def build_and_evaluate_model(params):
    """
    Builds and evaluates a Keras model based on the given parameters.
    
    Parameters:
        params (list): [num_layers, num_neurons, dropout_rate, learning_rate]
    
    Returns:
        tuple: (-validation_accuracy, training_time)
    """
    # Unpack parameters
    num_layers = int(params[0])  # Number of hidden layers
    num_neurons = int(params[1])  # Number of neurons per layer
    dropout_rate = params[2]  # Dropout rate
    learning_rate = params[3]  # Learning rate
    
    # Build the model
    model = Sequential()
    model.add(Dense(num_neurons, activation='relu', input_shape=(X_train.shape[1],)))
    for _ in range(num_layers - 1):
        model.add(Dense(num_neurons, activation='relu'))
        model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid'))  # Binary classification output
    
    # Compile the model
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    # Train the model and measure time
    start_time = time.time()
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, verbose=0, batch_size=32)
    training_time = time.time() - start_time
    
    # Evaluate model performance
    val_accuracy = history.history['val_accuracy'][-1]
    
    # Return objectives: minimize training time, maximize accuracy
    return -val_accuracy, training_time  # Negate accuracy for minimization

# 3. Hyperparameter Optimization Loop
from scipy.optimize import differential_evolution

# Define the parameter bounds: [num_layers, num_neurons, dropout_rate, learning_rate]
param_bounds = [
    (1, 5),         # Number of hidden layers
    (16, 128),      # Number of neurons per layer
    (0.0, 0.5),     # Dropout rate
    (1e-4, 1e-2)    # Learning rate
]
def optimization_function(params):
    accuracy, training_time = build_and_evaluate_model(params)
    # Combine objectives (adjust weights as needed)
    return accuracy + 0.1 * training_time  # Weight training time less heavily


# Run the optimization using Differential Evolution
result = differential_evolution(optimization_function, bounds=param_bounds, strategy='best1bin', 
                                 maxiter=3, popsize=5, seed=42, tol=0.01)

# result = differential_evolution(optimization_function, bounds=param_bounds, strategy='best1bin', 
#                                  maxiter=10, popsize=15, seed=42, tol=0.01)

# Extract the optimal parameters and objectives
optimal_params = result.x
optimal_objectives = result.fun

# 4. Final Model Training and Testing with Optimal Parameters
best_model_params = {
    'num_layers': int(optimal_params[0]),
    'num_neurons': int(optimal_params[1]),
    'dropout_rate': optimal_params[2],
    'learning_rate': optimal_params[3],
}

# Train the final model with optimal parameters
final_accuracy, final_training_time = build_and_evaluate_model(list(optimal_params))

# Print the results
print("\n=== Optimization Results ===")
print("Optimal Parameters Found:")
print(f" - Number of Layers: {int(optimal_params[0])}")
print(f" - Number of Neurons per Layer: {int(optimal_params[1])}")
print(f" - Dropout Rate: {optimal_params[2]:.2f}")
print(f" - Learning Rate: {optimal_params[3]:.4f}")
print("\n=== Final Model Performance ===")
print(f" - Final Validation Accuracy: {(-final_accuracy):.4f}")
print(f" - Total Training Time (seconds): {final_training_time:.2f}")

