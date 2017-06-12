# -*- coding: utf-8 -*-
"""
Created on Sat May 13 06:23:35 2017

This code is referring to the code from this link:
https://www.youtube.com/watch?v=d2BMirIToF4&list=PLXO45tsB95cI7ZleLM5i3XXhhe9YmVrRO&index=7
https://github.com/MorvanZhou/tutorials/blob/master/sklearnTUT/sk6_model_attribute_method.py

Python 3.5.2 |Anaconda 4.1.1 (64-bit)
Scikit-learn version: 0.17.1

"""

from __future__ import print_function
from sklearn import datasets
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt
import sklearn

print('*** scikit_tu03.py ***')
print('*** scikit_tu03 ***')
print('Scikit-learn version:', sklearn.__version__)


loaded_data = datasets.load_boston()
data_X = loaded_data.data
data_y = loaded_data.target

model = LinearRegression()
model.fit(data_X, data_y)

print(model.predict(data_X[:4, :]))
print(model.coef_) 
print(model.intercept_)
print(model.get_params())
print(model.score(data_X, data_y)) # R^2 coefficient of determination



