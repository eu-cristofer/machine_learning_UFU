#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 08:31:36 2024

@author: cristofer
"""

import pandas as pd
import seaborn as sns

home_data = pd.read_csv('data_files/housing.csv')
print(home_data.head())

import seaborn as sns

sns.scatterplot(
    data=home_data,
    x='longitude', 
    y='latitude', 
    hue='median_house_value'
)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(home_data[['latitude', 'longitude']], home_data[['median_house_value']], test_size=0.33, random_state=0)

from sklearn import preprocessing

X_train_norm = preprocessing.normalize(X_train)
X_test_norm = preprocessing.normalize(X_test)

from sklearn import KMeans

kmeans = KMeans(n_clusters = 3, random_state = 0, n_init='auto')
kmeans.fit(X_train_norm)

