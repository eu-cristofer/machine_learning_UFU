import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import time
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.operators.sampling.lhs import LatinHypercubeSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.core.callback import Callback
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
import matplotlib.pyplot as plt
import imageio

# 1. Preprocessing the data
data = pd.read_csv('data_files\\teste.csv')

X = data.drop(columns=['target'])
y = data['target']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# 2. Define the neural network model and evaluation function
def build_and_evaluate_model(params):
    num_layers = int(params[0])
    num_neurons = int(params[1])
    dropout_rate = params[2]
    learning_rate = params[3]

    model = Sequential()
    model.add(Dense(num_neurons, activation='relu', input_shape=(X_train.shape[1],)))
    for _ in range(num_layers - 1):
        model.add(Dense(num_neurons, activation='relu'))
        model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid'))

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    start_time = time.time()
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, verbose=0, batch_size=32)
    training_time = time.time() - start_time

    val_accuracy = history.history['val_accuracy'][-1]

    return -val_accuracy, training_time

# 3. Define the optimization problem
class NeuralNetworkOptimization(Problem):
    def __init__(self):
        super().__init__(
            n_var=4,
            n_obj=2,
            n_constr=0,
            xl=np.array([1, 16, 0.0, 1e-4]),
            xu=np.array([5, 128, 0.5, 1e-2])
        )

    def _evaluate(self, X, out, *args, **kwargs):
        f1 = []
        f2 = []
        for params in X:
            accuracy, training_time = build_and_evaluate_model(params)
            f1.append(accuracy)
            f2.append(training_time)
        out["F"] = np.column_stack([f1, f2])

# 4. Configure the NSGA-II algorithm
problem = NeuralNetworkOptimization()

algorithm = NSGA2(
    pop_size=20,
    sampling=LatinHypercubeSampling(),
    crossover=SBX(prob=0.9, eta=15),
    mutation=PolynomialMutation(eta=20),
    eliminate_duplicates=True
)

# Callback to capture intermediate results
class MyCallback(Callback):
    def __init__(self):
        super().__init__()
        self.data["F"] = []

    def notify(self, algorithm):
        self.data["F"].append(algorithm.pop.get("F"))

callback = MyCallback()

# 5. Run the optimization
result = minimize(
    problem,
    algorithm,
    ('n_gen', 20),
    seed=42,
    verbose=True,
    callback=callback
)

# 6. Visualization with different colors
non_dominated = NonDominatedSorting().do(result.F)

# Create a mask for Pareto front
pareto_mask = np.zeros(result.F.shape[0], dtype=bool)
pareto_mask[non_dominated] = True

# Separate Pareto front and non-Pareto front points
pareto_front = result.F[pareto_mask]
non_pareto = result.F[~pareto_mask]

# Create a scatter plot with Matplotlib
plt.figure(figsize=(10, 6))
plt.scatter(pareto_front[:, 0], pareto_front[:, 1], color="blue", label="Pareto Front")
plt.scatter(non_pareto[:, 0], non_pareto[:, 1], color="red", label="Other Points")
plt.xlabel("Acurácia (Negativa)")
plt.ylabel("Tempo de Treinamento (s)")
plt.title("Fronte de Pareto: Acurácia vs. Tempo de Treinamento")
plt.legend()
plt.grid(True)
plt.show()

# 7. Generate a GIF of the training process
frames = []
for gen, front in enumerate(callback.data["F"]):
    plt.figure()
    plt.scatter(front[:, 0], front[:, 1], label=f"Generation {gen}", color="green")
    plt.xlabel("Acurácia (Negativa)")
    plt.ylabel("Tempo de Treinamento (s)")
    plt.title(f"Evolution of Pareto Front - Generation {gen}")
    plt.legend()
    plt.grid(True)

    # Save frame to the GIF
    plt.savefig(f"gen_{gen}.png")
    frames.append(f"gen_{gen}.png")
    plt.close()

# Create the GIF
with imageio.get_writer("pareto_evolution.gif", mode="I", duration=0.5) as writer:
    for frame in frames:
        image = imageio.imread(frame)
        writer.append_data(image)

# Cleanup temporary images
import os
for frame in frames:
    os.remove(frame)
