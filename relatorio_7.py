#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 20:38:47 2024

Seleção das 12 melhores features para o banco de dados otto_group.csv

@author: Cristofer Antoni Souza Costa
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import (SelectKBest,
                                       f_classif,
                                       RFE,
                                       mutual_info_classif,
                                       SequentialFeatureSelector)
from sklearn.linear_model import (Lasso,
                                  LinearRegression)
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from collections import Counter

# Load the Otto Group dataset
data = pd.read_csv("data_files/otto_group.csv")

# Separate features and target variable
X_raw = data.drop(columns=['id', 'target'])
X = X_raw.values
y = data['target']

# Encode the target variable if it's categorical
if y.dtype == 'object':
    y = LabelEncoder().fit_transform(y)

# Set the number of top features to select
k = 12

# Define maximum iterations as a variable for easy adjustment
max_iter = 1000

# Dictionary to store selected features from each metric
selected_features = {}

# Function to print and store selected features
def add_selected_features(method_name, feature_indices):
    features = X_raw.columns[feature_indices]
    selected_features[method_name] = features.tolist()
    print(f"{method_name}")
    print(features.tolist(), end='\n\n')

# 1. Univariate feature selection using ANOVA F-test for classification
selector = SelectKBest(score_func=f_classif, k=k)
selector.fit(X, y)
add_selected_features("Univariate Selection (ANOVA F-test)",
                      selector.get_support(indices=True))

# 2. Recursive Feature Elimination (RFE) with Linear Regression
model = LinearRegression()
rfe = RFE(estimator=model, n_features_to_select=k)
rfe.fit(X, y)
rfe_indices = rfe.get_support(indices=True)
add_selected_features("Recursive Feature Elimination (RFE)",
                      rfe_indices)

# 3. L1 Regularization (Lasso)
lasso = Lasso(alpha=0.01, max_iter=max_iter)
lasso.fit(X, y)
lasso_indices = np.argsort(np.abs(lasso.coef_))[-k:]  # select indices of top k features
add_selected_features("L1 Regularization (Lasso)",
                      lasso_indices)

# 4. Tree-Based Methods (Random Forest)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)
rf_indices = np.argsort(rf.feature_importances_)[-k:]  # select indices of top k features
add_selected_features("Random Forest Feature Importance",
                      rf_indices)

# 5. Principal Component Analysis (PCA)
pca = PCA(n_components=k)
X_pca = pca.fit_transform(X)
pca_features = np.argsort(np.abs(pca.components_).sum(axis=0))[-k:]  # top k contributing original features
add_selected_features("Principal Component Analysis (PCA)",
                      pca_features)

# 6. Correlation-based Feature Selection
corr_scores = X_raw.corrwith(pd.Series(y))
corr_indices = np.argsort(corr_scores.abs())[-k:]
add_selected_features("Correlation-Based Selection",
                      corr_indices)

# 7. Mutual Information
mi_scores = mutual_info_classif(X, y)
mi_indices = np.argsort(mi_scores)[-k:]
add_selected_features("Mutual Information",
                      mi_indices)

# 8. Sequential Feature Selection with Linear Regression
sfs = SequentialFeatureSelector(LinearRegression(),
                                n_features_to_select=k,
                                direction='forward')
sfs.fit(X, y)
add_selected_features("Sequential Feature Selection (SFS)",
                      sfs.get_support(indices=True))

# Frequency Counter
feature_counts = Counter()
for method, features in selected_features.items():
    feature_counts.update(features)

# Rank
ranked_features = feature_counts.most_common(k)
top_features, frequencies = zip(*ranked_features)

# Print the top features ranked by frequency across methods
print("\nTop 12 Features Across All Methods (Ranked by Frequency):")
for i, (feature, freq) in enumerate(ranked_features, 1):
    print(f"{i}. {feature}\t(Selected {freq} times)")

# Plot the top features by frequency
plt.figure(figsize=(10, 6))
plt.bar(top_features, frequencies)
plt.xlabel("Features")
plt.ylabel("Frequency of Selection Across Methods")
plt.title("Top 12 Features Ranked by Selection Frequency Across Methods")
plt.xticks(rotation=90)
plt.show()

# Summary plot
ordered_features = X_raw.columns
plt.figure(figsize=(15, 6))
for method, features in selected_features.items():
    feature_positions = [ordered_features.get_loc(f) for f in features]
    plt.plot(feature_positions, [method] * len(features), 'o', label=method)
plt.xlabel("Feature Index (Original Order)")
plt.ylabel("Feature Selection Methods")
plt.title("Summary of Selected Features Across Methods (Original Feature Order)")
plt.xticks(ticks=range(len(ordered_features)),
           labels=ordered_features,
           rotation=90)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()
