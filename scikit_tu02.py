# -*- coding: utf-8 -*-
"""
Created on Sat May 13 05:28:56 2017

This code is referring to the code from this link:
https://www.youtube.com/watch?v=EvV99YhSsJU&index=5&list=PLXO45tsB95cI7ZleLM5i3XXhhe9YmVrRO
https://github.com/MorvanZhou/tutorials/blob/master/sklearnTUT/sk4_learning_pattern.py

Python 3.5.2 |Anaconda 4.1.1 (64-bit)
Scikit-learn version: 0.17.1

"""

from __future__ import print_function
from sklearn import datasets
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt
import sklearn

print('*** scikit_tu02.py ***')
print('*** scikit_tu02 ***')
print('Scikit-learn version:', sklearn.__version__)


loaded_data = datasets.load_boston()
data_X = loaded_data.data
data_y = loaded_data.target

model = LinearRegression()
model.fit(data_X, data_y)

print(model.predict(data_X[:4, :]))
print(data_y[:4])

X, y = datasets.make_regression(n_samples=100, n_features=1, n_targets=1, noise=10)
plt.scatter(X, y)
plt.show()

