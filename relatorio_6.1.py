# -*- coding: utf-8 -*-
"""
Título: Estudo Dirigido: Comparando técnicas de aprendizado de máquina usando pipelines

Autor: Cristofer Antoni Souza Costa
Email: cristofercosta@yahoo.com.br
"""

import pandas as pd
import numpy as np
import tkinter.filedialog
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn import svm, tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

# Prompt user to select the file
filename = tkinter.filedialog.askopenfilename(
    title="Escolha o Arquivo",
    filetypes=[
        ("Arquivos CSV", "*.csv"),
        ("Arquivos Excel", "*.xlsx"),
        ("Todos Arquivos", "*.*"),
    ],
)
if not filename:
    print("Ending the session.")
    exit(0)

# Load the dataset
data = pd.read_csv(filename)

# Histogram plot
feat = data.loc[:, "feat_1":"feat_30"]
plt.figure(figsize=(30, 20))
feat.hist(figsize=(30, 20))
plt.suptitle("Histogram of Features", fontsize=16)
plt.show()

# Boxplot
plt.figure(figsize=(10, 10))
feat.boxplot()
plt.title("Boxplot of Features")
plt.show()

# Check for missing data
missing_data = feat.isna().any().any()
print(f"Is there any missing data? {missing_data}")

# Heatmap plot
feat = data.loc[:, "feat_1":"feat_93"]
plt.figure(figsize=(12, 10))
sns.heatmap(feat.corr(), center=0, cmap="coolwarm", annot=True, fmt=".2f")
plt.title("Heatmap of Feature Correlations")
plt.show()

# Precompute PCA (Speeds up pipeline)
pca = PCA(n_components=27)
X_train_pca = pca.fit_transform(data.loc[:, "feat_1":"feat_93"])
X_test_pca = pca.transform(data.loc[:, "feat_1":"feat_93"])

# Prepare target variable
y = data["target"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X_train_pca, y, test_size=0.4, random_state=42
)

# Random Forest Grid Search (Parallel processing enabled)
param_grid_forest = {
    "n_estimators": [50, 100],
    "criterion": ["gini"],
    "max_depth": [5],
    "min_samples_leaf": [0.1],
    "min_samples_split": [0.1],
}

rf = RandomForestClassifier(
    random_state=123, warm_start=True
)  # Warm start optimization
gs_rf = GridSearchCV(
    rf, param_grid=param_grid_forest, scoring="accuracy", cv=3, verbose=2, n_jobs=-1
)
gs_rf.fit(X_train, y_train)
print(f"Random Forest Best Accuracy: {gs_rf.best_score_:.3f}")
print("Best Params:", gs_rf.best_params_)

# AdaBoost Randomized Search (Speeds up by sampling parameter space)
adaboost_param_grid = {"n_estimators": [30, 50, 70], "learning_rate": [1.0, 0.5, 0.1]}

ab = AdaBoostClassifier(random_state=123)
random_search_ab = RandomizedSearchCV(
    ab,
    param_distributions=adaboost_param_grid,
    n_iter=5,
    scoring="accuracy",
    cv=3,
    verbose=2,
    n_jobs=-1,
)
random_search_ab.fit(X_train, y_train)
print(f"AdaBoost Best Accuracy: {random_search_ab.best_score_:.3f}")
print("Best Params:", random_search_ab.best_params_)

# SVM Grid Search (Parallel processing enabled)
param_grid_svm = [
    {"C": [0.1, 1], "kernel": ["linear"]},
    {"C": [1], "gamma": [0.001, 0.01], "kernel": ["rbf"]},
]

svm_model = svm.SVC(random_state=123)
gs_svm = GridSearchCV(
    svm_model, param_grid=param_grid_svm, scoring="accuracy", cv=3, verbose=2, n_jobs=-1
)
gs_svm.fit(X_train, y_train)
print(f"SVM Best Accuracy: {gs_svm.best_score_:.3f}")
print("Best Params:", gs_svm.best_params_)

# Logistic Regression Pipeline (Additional Model)
logistic = LogisticRegression(random_state=123)
logistic.fit(X_train, y_train)
logistic_accuracy = logistic.score(X_test, y_test)
print(f"Logistic Regression Accuracy: {logistic_accuracy:.3f}")

# Decision Tree Pipeline (Additional Model)
decision_tree = tree.DecisionTreeClassifier(random_state=123)
decision_tree.fit(X_train, y_train)
tree_accuracy = decision_tree.score(X_test, y_test)
print(f"Decision Tree Accuracy: {tree_accuracy:.3f}")


# Print the entire cv_results_ dictionary (can be large)
print(gs_rf.cv_results_)