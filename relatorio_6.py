# -*- coding: utf-8 -*-
"""
Título: Estudo Dirigido: Comparando técnicas de aprendizado de máquina
usando pipelines 

Autor: Cristofer Antoni Souza Costa
Email: cristofercosta@yahoo.com.br
"""

import pandas as pd
import numpy as np
import tkinter.filedialog
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn import svm, tree
from sklearn.ensemble import RandomForestClassifier

# Prompt user to select the file
filename = tkinter.filedialog.askopenfilename(
    title="Escolha o Arquivo",
    filetypes=[("Arquivos CSV", "*.csv"), ("Arquivos Excel", "*.xlsx"), ("Todos Arquivos", "*.*")]
)
if not filename:
    print("Ending the session.")
    exit(0)

# Load the dataset
data = pd.read_csv(filename)

# Histogram plot
feat = data.loc[:, 'feat_1':'feat_30']
plt.figure(figsize=(30, 20))
feat.hist(figsize=(30, 20))
plt.suptitle('Histogram of Features', fontsize=16)
plt.show()

# Boxplot
plt.figure(figsize=(10, 10))
feat.boxplot()
plt.title('Boxplot of Features')
plt.show()

# Check for missing data
missing_data = feat.isna().any().any()
print(f"Is there any missing data? {missing_data}")

# Heatmap plot
feat = data.loc[:, 'feat_1':'feat_93']
plt.figure(figsize=(12, 10))
sns.heatmap(feat.corr(), center=0, cmap='coolwarm', annot=True, fmt='.2f')
plt.title('Heatmap of Feature Correlations')
plt.show()

# Perform PCA with different components
pca_components = [20, 40, 60]
for n in pca_components:
    pca = PCA(n_components=n)
    principal_components = pca.fit_transform(feat)
    explained_variance = np.sum(pca.explained_variance_ratio_)
    print(f"PCA with {n} components - Explained variance: {explained_variance:.2f}")

# Final PCA with 27 components
pca_final = PCA(n_components=27)
principal_components_final = pca_final.fit_transform(feat)
print(f"Final PCA explained variance: {np.sum(pca_final.explained_variance_ratio_):.2f}")

# Heatmap of PCA-transformed data
plt.figure(figsize=(12, 10))
sns.heatmap(pd.DataFrame(principal_components_final).corr(), center=0, cmap='coolwarm', annot=True, fmt='.2f')
plt.title('Heatmap of PCA-Transformed Data')
plt.show()

# Prepare data for modeling
y = data['target']
X = data.loc[:, 'feat_1':'feat_93']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Construct pipeline for Random Forest
pipe_rf = Pipeline([
    ('pca', PCA(n_components=27)),
    ('clf', RandomForestClassifier(random_state=123))
])

# Set grid search parameters
param_grid_forest = [
    {
        'clf__n_estimators': [120],
        'clf__criterion': ['entropy', 'gini'],
        'clf__max_depth': [4, 5, 6],
        'clf__min_samples_leaf': [0.05, 0.1, 0.2],
        'clf__min_samples_split': [0.05, 0.1, 0.2]
    }
]

# Construct GridSearchCV
gs_rf = GridSearchCV(
    estimator=pipe_rf,
    param_grid=param_grid_forest,
    scoring='accuracy',
    cv=3,
    verbose=2,
    return_train_score=True
)

# Fit using grid search
gs_rf.fit(X_train, y_train)

# Best accuracy and parameters
print('Best accuracy: %.3f' % gs_rf.best_score_)
print('\nBest params:\n', gs_rf.best_params_)

# Construct and evaluate additional pipelines
pipelines = [
    Pipeline([('pca', PCA(n_components=27)), ('clf', LogisticRegression(random_state=123))]),
    Pipeline([('pca', PCA(n_components=27)), ('clf', svm.SVC(random_state=123))]),
    Pipeline([('pca', PCA(n_components=27)), ('clf', tree.DecisionTreeClassifier(random_state=123))])
]

pipeline_names = ['Logistic Regression', 'Support Vector Machine', 'Decision Tree']

# Fit and evaluate each pipeline
for name, pipe in zip(pipeline_names, pipelines):
    pipe.fit(X_train, y_train)
    accuracy = pipe.score(X_test, y_test)
    print(f'{name} pipeline test accuracy: {accuracy:.3f}')

gs_rf.cv_results_
