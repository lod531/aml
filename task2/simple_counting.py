# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import SVC


# Importing the dataset
dataset_X_train = pd.read_csv('X_train.csv')
dataset_y_train = pd.read_csv('y_train.csv')
X_train = dataset_X_train.iloc[:,1:].values
y_train = dataset_y_train.iloc[:,1].values

dataset_X_test = pd.read_csv('X_test.csv')
ids_test = dataset_X_test.iloc[:,0].values
X_test = dataset_X_test.iloc[:,1:].values

counts = [0, 0, 0]
for i in y_train:
    counts[int(i)] += 1

total = sum(counts)
print('count:', counts)
print('percentages')
for i, count in enumerate(counts):
    print(i, '%:', count/total)
