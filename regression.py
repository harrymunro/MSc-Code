# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 10:40:11 2016

@author: harrymunro

Linear regression testing boarders and alighters vs mean dwell time.
"""

from sklearn import linear_model
from sklearn.externals import joblib
import numpy as np
import pandas as pd

filename = 'resampled_by_unique_location.csv'
df = pd.read_csv(filename)

y = np.array(df['DWELL TIME'])
x1 = df['BOARDERS AND ALIGHTERS']
x2 = df['SAF RATE']
x3 = df['SEATING CAPACITY']
x4 = df['STANDING CAPACITY']
x5 = df['TRAIN DENSITY']

X = np.array([x1, x2, x3, x4, x5])
#X = np.array([x3])
X = X.T

# lasoo linear regression
reg = linear_model.Lasso(alpha = 0.1)
reg.fit(X, y)
lassoo_score = reg.score(X, y)

# ordinary linear regression
reg = linear_model.LinearRegression()
reg.fit(X, y)
score_ordinary_linear = reg.score(X, y)
params_ordinary_linear = reg.get_params()
joblib.dump(reg, 'linear_regression_model.pkl')

from scipy import stats
#slope, intercept, r_value, p_value, std_err = stats.linregress(df['BOARDERS AND ALIGHTERS'],df['DWELL TIME'])
#r2_linear = r_value**2

# decision tree
from sklearn import tree
reg = tree.DecisionTreeRegressor()
reg = reg.fit(X, y)
tree_score = reg.score(X, y)
for n in range(2, 51, 2):
    reg = tree.DecisionTreeRegressor(max_depth = n)
    reg = reg.fit(X, y)
    print(reg.score(X, y))

# neural network
from sklearn import neural_network
reg = neural_network.MLPRegressor(hidden_layer_sizes = 100)
reg = reg.fit(X, y)
score_nn = reg.score(X, y)
params_nn = reg.get_params()
for n in range(10, 310, 10):
    reg = neural_network.MLPRegressor(hidden_layer_sizes = n)
    reg = reg.fit(X, y)
    print(reg.score(X, y))

# SGD
reg = linear_model.SGDRegressor()
reg = reg.fit(X, y)
score_sgd = reg.score(X, y)

# SVM
from sklearn import svm
reg = svm.SVR()
reg = reg.fit(X, y)
score_svm = reg.score(X, y)

# KNN
from sklearn.neighbors import KNeighborsRegressor
neigh = KNeighborsRegressor()
reg = neigh.fit(X, y)
reg.score(X,y)

for n in range(2, 21):
    neigh = KNeighborsRegressor(n_neighbors = n, weights = 'distance')
    reg = neigh.fit(X, y)
    s = reg.score(X,y)
    print("%r" % s)
