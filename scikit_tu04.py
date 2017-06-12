# -*- coding: utf-8 -*-
"""
Created on Sat May 13 07:36:57 2017

This code is referring to the code from this link:
https://www.youtube.com/watch?v=3GxT8n0ShsU&list=PLXO45tsB95cI7ZleLM5i3XXhhe9YmVrRO&index=8
https://github.com/MorvanZhou/tutorials/blob/master/sklearnTUT/sk7_normalization.py

Python 3.5.2 |Anaconda 4.1.1 (64-bit)
Scikit-learn version: 0.17.1

"""

from __future__ import print_function
from sklearn import preprocessing
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.datasets.samples_generator import make_classification
from sklearn.svm import SVC

import matplotlib.pyplot as plt
import sklearn

print('*** scikit_tu04.py ***')
print('*** scikit_tu04 ***')
print('Scikit-learn version:', sklearn.__version__)



a = np.array([[10, 2.7, 3.6],
                     [-100, 5, -2],
                     [120, 20, 40]], dtype=np.float64)
print(a)
print(preprocessing.scale(a))

X, y = make_classification(n_samples=300, n_features=2 , n_redundant=0, n_informative=2,
                           random_state=22, n_clusters_per_class=1, scale=100)
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()
X = preprocessing.scale(X)    # normalization step
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)
clf = SVC()
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))


