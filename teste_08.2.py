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
from pymoo.visualization.scatter import Scatter

# 1. Pré-processamento dos dados
data = pd.read_csv('data_files\\teste.csv')

X = data.drop(columns=['target'])
y = data['target']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# 2. Definição do modelo de rede neural e função de avaliação
def build_and_evaluate_model(params):
    num_layers = int(params[0])  # Número de camadas ocultas
    num_neurons = int(params[1])  # Número de neurônios por camada
    dropout_rate = params[2]  # Taxa de dropout
    learning_rate = params[3]  # Taxa de aprendizado
    
    model = Sequential()
    model.add(Dense(num_neurons, activation='relu', input_shape=(X_train.shape[1],)))  # Primeira camada
    for _ in range(num_layers - 1):
        model.add(Dense(num_neurons, activation='relu'))
        model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid'))  # Saída para classificação binária
    
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    start_time = time.time()
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, verbose=0, batch_size=32)
    training_time = time.time() - start_time
    
    val_accuracy = history.history['val_accuracy'][-1]
    
    return -val_accuracy, training_time  # Negar acurácia para minimização

# 3. Definição do problema de otimização
class NeuralNetworkOptimization(Problem):
    def __init__(self):
        super().__init__(
            n_var=4,  # Número de variáveis (hiperparâmetros)
            n_obj=2,  # Dois objetivos: minimizar acurácia negativa e tempo de treinamento
            n_constr=0,  # Sem restrições
            xl=np.array([1, 16, 0.0, 1e-4]),  # Limites inferiores
            xu=np.array([5, 128, 0.5, 1e-2])  # Limites superiores
        )
    
    def _evaluate(self, X, out, *args, **kwargs):
        f1 = []
        f2 = []
        for params in X:
            accuracy, training_time = build_and_evaluate_model(params)
            f1.append(accuracy)
            f2.append(training_time)
        out["F"] = np.column_stack([f1, f2])

# 4. Configuração do algoritmo NSGA-II
problem = NeuralNetworkOptimization()

algorithm = NSGA2(
    pop_size=20,
    sampling=LatinHypercubeSampling(),  # Usando Latin Hypercube para melhor amostragem inicial
    crossover=SBX(prob=0.9, eta=15),  # Crossover binomial simulado
    mutation=PolynomialMutation(eta=20),  # Mutação polinomial
    eliminate_duplicates=True
)

# 5. Execução da otimização
result = minimize(
    problem,
    algorithm,
    ('n_gen', 10),  # Número de gerações
    seed=42,
    verbose=True
)

# 6. Análise dos resultados
pareto_front = result.F  # Objetivos no fronte de Pareto
pareto_solutions = result.X  # Parâmetros correspondentes

# 7. Visualização do fronte de Pareto
plot = Scatter(title="Fronte de Pareto: Acurácia vs. Tempo de Treinamento")
plot.add(pareto_front, facecolor="blue", edgecolor="black")
plot.show()

# 8. Exibição das melhores soluções com identificação dos parâmetros
print("\n=== Soluções do Fronte de Pareto ===")
parameter_names = ['Número de Camadas Ocultas', 'Número de Neurônios por Camada', 'Taxa de Dropout', 'Taxa de Aprendizado']
for i, (accuracy, time) in enumerate(pareto_front):
    print(f"\nSolução {i+1}:")
    print(f" - Acurácia (Negativa): {-accuracy:.4f}")
    print(f" - Tempo de Treinamento (s): {time:.2f}")
    print(" - Parâmetros:")
    for name, value in zip(parameter_names, pareto_solutions[i]):
        print(f"    {name}: {value:.4f}")
