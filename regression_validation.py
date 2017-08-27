#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 13:13:29 2017

@author: Harry

- Take in a dataset
- Partition into training and test data
- Fit regression model to training set
- Create predictions for test set
- Calculate peareson correlation coeffecient
"""

import numpy as np
import pandas as pd
import scipy.stats as stats

filename = 'resampled_by_unique_location.csv'
df = pd.read_csv(filename)
df = df.sample(frac = 1) # randomise order of dataset
df_train = df.head(n = int(len(df) * 0.8))
df_test = df.tail(n = int(len(df) * 0.2))

# training time
y_train = np.array(df_train['DWELL TIME'])
x1_train = df_train['BOARDERS AND ALIGHTERS']
x2_train = df_train['SAF RATE']
x3_train = df_train['SEATING CAPACITY']
x4_train = df_train['STANDING CAPACITY']
x5_train = df_train['TRAIN DENSITY']

X_train = np.array([x1_train, x2_train, x3_train, x4_train, x5_train])
X_train = X_train.T

# testing time
y_test = np.array(df_test['DWELL TIME'])
x1_test = df_test['BOARDERS AND ALIGHTERS']
x2_test = df_test['SAF RATE']
x3_test = df_test['SEATING CAPACITY']
x4_test = df_test['STANDING CAPACITY']
x5_test = df_test['TRAIN DENSITY']

X_test = np.array([x1_test, x2_test, x3_test, x4_test, x5_test])
X_test = X_test.T

# linear regression model
from sklearn import linear_model
reg = linear_model.LinearRegression()
reg.fit(X_train, y_train)
score_ordinary_linear = reg.score(X_train, y_train)

predictions = []
for row in X_test:
    pred = reg.predict(row)
    predictions.append(pred[0])

y = []
for n in y_test:
    y.append(n)
    
pearson_corr_linear_regression = stats.pearsonr(y, predictions)   

# decision tree

print("\nDECISION TREE RESULTS BELOW")
from sklearn import tree
reg = tree.DecisionTreeRegressor()
tree_scores = []
tree_correlations = []
for n in range(2, 51, 1):
    reg = tree.DecisionTreeRegressor(max_depth = n)
    reg = reg.fit(X_train, y_train)
    tree_score = reg.score(X_train, y_train)
    predictions = []
    for row in X_test:
        pred = reg.predict(row)
        predictions.append(pred[0])
    pearson_corr_decision_tree = stats.pearsonr(y_test, predictions)
    tree_scores.append(tree_score)
    tree_correlations.append(pearson_corr_decision_tree)
print("\nR-SQUARED SCORES")
print(tree_scores)
print("\nCORRELATION SCORES")
print(tree_correlations)


# K Nearest Neighbours

from sklearn.neighbors import KNeighborsRegressor
neigh = KNeighborsRegressor()

knn_scores = []
knn_correlations = []
for n in range(2, 41):
    neigh = KNeighborsRegressor(n_neighbors = n, weights = 'uniform')
    reg = neigh.fit(X_train, y_train)
    s = reg.score(X_train,y_train)
    knn_scores.append(s)
    predictions = []
    for row in X_test:
        pred = reg.predict(row)
        predictions.append(pred[0])
    pearson_corr_knn = stats.pearsonr(y_test, predictions)
    knn_correlations.append(pearson_corr_knn)
    
#for score in knn_correlations:
   # print(score[0])
   
# neural network
from sklearn import neural_network
nn_scores = []
nn_correlations = []
for n in range(1, 51):
    reg = neural_network.MLPRegressor(hidden_layer_sizes = n)
    reg = reg.fit(X_train, y_train)
    nn_score = reg.score(X_train, y_train)
    nn_scores.append(nn_score)
    predictions = []
    for row in X_test:
        pred = reg.predict(row)
        predictions.append(pred[0])
    pearson_corr_nn = stats.pearsonr(y_test, predictions)
    nn_correlations.append(pearson_corr_nn)
    