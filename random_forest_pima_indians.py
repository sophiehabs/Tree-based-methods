# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 21:00:31 2019

@author: Pablo
"""

# Random Forest Classification
import pandas
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pandas.read_csv(url, names=names)
array = dataframe.values

X = array[:,0:8]
Y = array[:,8]

seed = 7
num_trees = 100
max_features = 3
kfold = model_selection.KFold(n_splits=10)

model = RandomForestClassifier(n_estimators=num_trees, max_features=max_features)

results = model_selection.cross_val_score(model, X, Y, cv=kfold)
print("Random Forest Accuracy : ",results.mean())