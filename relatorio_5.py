# -*- coding: utf-8 -*-
"""
Título: Métodos de Validação Cruzada para Classificação SVM no Conjunto de Dados de Câncer

Autor: Cristofer Antoni Souza Costa
Email: cristofercosta@yahoo.com.br

Descrição:
Este script implementa alguns métodos de validação cruzada para avaliar o desempenho 
de um classificador SVM (Support Vector Machine) utilizando o kernel RBF no conjunto
de dados de câncer. 

Os métodos aplicados incluem Validação Cruzada K-Fold, K-Fold Estratificado e Monte
Carlo (ShuffleSplit). 

Além disso, um método de Bootstrap é utilizado para estimar intervalos de confiança
para o desempenho do modelo no conjunto de teste.

O conjunto de dados é carregado a partir de um arquivo CSV e características específicas
são selecionadas para a tarefa de classificação.

Antes do treinamento, as características são normalizadas para garantir que todas estejam
na mesma escala.

As técnicas de validação cruzada garantem a generalização do modelo, dividindo os dados
em múltiplos subconjuntos de treinamento e teste, enquanto o método de Bootstrap fornece
um intervalo de confiança para a avaliação do modelo.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold, ShuffleSplit
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler



def Bootstrap(model, x_test, y_test):
    """
    Applies the Bootstrap resampling technique to estimate the confidence interval
    for model performance (accuracy) on the test set.
    
    Parameters:
    - model: Trained classifier model
    - x_test: Test set features
    - y_test: True labels of the test set
    
    Returns:
    - IC: 90% Confidence Interval for the model's accuracy on the test set
    """
    N = 500 
    n = len(y_test)
    result = np.zeros(500) 
    for i in range(N):
        I = np.random.choice(n, n, replace=True) 
        result[i] = model.score(x_test[I, :], y_test.iloc[I]) 
    IC = np.quantile(result, [.05, .95])
    return IC

# Carregando o conjunto de dados
data = pd.read_csv(r"C:\Users\EMKA\OneDrive - PETROBRAS\10 - GitHub\machine_learning_UFU\data_files\Cancer_Data.csv")

# Seleção de features
features = ['radius_worst',
            'concave points_worst',
            'perimeter_se',
            'texture_worst']

# Dividindo os dados em features (X) e target (y)
y = data['diagnosis']
x = data[features]

# Normalização dos dados
scaler = StandardScaler()
x = scaler.fit_transform(x)

# Divisão treino-teste (70% para treino, 30% para teste)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=None)

# Configuração do modelo SVM com kernel RBF
model = SVC(kernel='rbf')

# Treinando o modelo SVM
model.fit(x_train, y_train)

# Avaliando o modelo nos conjuntos de treino e teste
train_score = model.score(x_train, y_train)  # Acurácia no conjunto de treino
test_score = model.score(x_test, y_test)     # Acurácia no conjunto de teste
print('Desempenho do modelo SVM com kernel RBF:')
print(f'  Score do conjunto de treino = {train_score:.2f}')
print(f'  Score do conjunto de teste = {test_score:.2f}')

# Validação Cruzada K-Fold (10 dobras)
kfold_validation = KFold(10)
cv_scores_kfold = cross_val_score(model, x_train, y_train, cv=kfold_validation)
print('Validação Cruzada K-Fold:')
print(f'  Média = {np.mean(cv_scores_kfold):.2f}')
print(f'  Desvio Padrão = {np.std(cv_scores_kfold):.2f}')

# Intervalo de Confiança Bootstrap (90% IC)
IC = Bootstrap(model, x_test, y_test)
print('Intervalo de Confiança de 90% para o conjunto de teste')
print(f'  [{IC[0]:.2f}, {IC[1]:.2f}]')

# Validação Cruzada K-Fold Estratificada (5 dobras)
sk_fold = StratifiedKFold(n_splits=5)
cv_scores_stratified = cross_val_score(model, x, y, cv=sk_fold)
print('K-Fold Estratificado:')
print(f'  Média = {np.mean(cv_scores_stratified):.2f}')
print(f'  Desvio Padrão = {np.std(cv_scores_stratified):.2f}')

# Validação Monte Carlo (ShuffleSplit, 10 iterações, 70% treino, 30% teste)
shuffle_split = ShuffleSplit(n_splits=10, test_size=0.30)
cv_scores_shuffle = cross_val_score(model, x, y, cv=shuffle_split)
print('Monte Carlo (ShuffleSplit):')
print(f'  Média = {np.mean(cv_scores_shuffle):.2f}')
print(f'  Desvio Padrão = {np.std(cv_scores_shuffle):.2f}')