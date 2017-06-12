# -*- coding: utf-8 -*-
"""
Created on Fri May 12 06:20:03 2017

This code is referring to the code from this link:
https://www.youtube.com/watch?v=EvV99YhSsJU&index=5&list=PLXO45tsB95cI7ZleLM5i3XXhhe9YmVrRO
https://github.com/MorvanZhou/tutorials/blob/master/sklearnTUT/sk4_learning_pattern.py

Python 3.5.2 |Anaconda 4.1.1 (64-bit)
Scikit-learn version: 0.17.1

"""

from __future__ import print_function
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier


import numpy as np
import sklearn





print('*** scikit_tu01.py ***')
print('*** scikit_tu01 ***')
print('Scikit-learn version:', sklearn.__version__)




iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target

##print(iris_X[:2, :])
##print(iris_y)

X_train, X_test, y_train, y_test = train_test_split(
    iris_X, iris_y, test_size=0.3)

##print(y_train)

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
print(knn.predict(X_test))
print(y_test)


