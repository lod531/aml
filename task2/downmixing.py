# ---------------------------------- TASK 2 ----------------------------------

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset_X_train = pd.read_csv('X_train.csv')
dataset_y_train = pd.read_csv('y_train.csv')
X_train = dataset_X_train.iloc[:,1:].values
y_train = dataset_y_train.iloc[:,1].values

dataset_X_test = pd.read_csv('X_test.csv')
ids_test = dataset_X_test.iloc[:,0].values
X_test = dataset_X_test.iloc[:,1:].values

# Dealing with imbalanced classes

from sklearn.utils import resample
l = []
i = 0
while i < len(y_train):
     if y_train[i] == 0:
         l = l+[i]
     i = i+1
     

# 600 etichettate 0
     
m = []
i = 0
while i < len(y_train):
     if y_train[i] == 1:
         m = m+[i]
     i = i+1 


# 3600 etichettate 1
     
n = []
i = 0
while i < len(y_train):
     if y_train[i] == 2:
         n = n+[i]
     i = i+1   


# 600 etichettate 2
    
# Downsampling

y_0 = y_train[l]
y_1 = y_train[m]
y_1 = y_1[0:900]
y_2 = y_train[n]
majority_class_1 = X_train[m]

class_1 = resample(majority_class_1, 
                                 replace=False,    # sample without replacement
                                 n_samples=900,     # to match minority class
                                 random_state=123)
  
class_2 = X_train[n]
class_0 = X_train[l]
X_train_new = np.append(class_0, class_1, axis = 0)
X_train = np.append(X_train_new, class_2, axis = 0)
y_0 = y_0.reshape(-1,1)
y_1 = y_1.reshape(-1,1)
y_2 = y_2.reshape(-1,1)
y_train_new = np.append(y_0, y_1, axis = 0)
y_train = np.append(y_train_new, y_2, axis = 0)
"""
# Oversampling

# Creating y_train
y_0 = y_train[l]

y_0 = y_0.reshape(-1,1)
y_00 = np.append(y_0,y_0,axis=0)
y_000 = np.append(y_00,y_00,axis=0)
y_0 = np.append(y_00, y_000, axis = 0)


y_1 = y_train[m]
y_1 = y_1.reshape(-1,1)

y_2 = y_train[n]
y_2 = y_2.reshape(-1,1)
y_22 = np.append(y_2,y_2,axis=0)
y_222 = np.append(y_22,y_22,axis=0)
y_2 = np.append(y_22, y_222, axis = 0)

y_train_new = np.append(y_0, y_1, axis = 0)
y_train = np.append(y_train_new, y_2, axis = 0)


# Creating X_train
class_2 = X_train[n]
class_0 = X_train[l]
class_1 = X_train[m]
class_0 = resample(class_0, 
                                replace=True,     # sample with replacement
                                n_samples=3600,    # to match majority class
                                random_state=123)
class_2 = resample(class_2, 
                                replace=True,     # sample with replacement
                                n_samples=3600,    # to match majority class
                                random_state=123)

X_train_new = np.append(class_0, class_1, axis = 0)
X_train = np.append(X_train_new, class_2, axis = 0)
"""

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# One-Vs-One
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import SVC
classifier = SVC(C = 1,
                 class_weight="balanced",
                 kernel = 'rbf', 
                 random_state = 0,
                 gamma = 0.967e-3,
                 decision_function_shape = 'ovo')

classifier.fit(X_train, y_train)
y_test_pred = classifier.predict(X_test)
sol = np.append(arr = ids_test.reshape(-1,1), values = y_test_pred.reshape(-1,1), axis = 1)
fsol = pd.DataFrame(sol)
fsol.rename(columns={0: 'id', 1: 'y'}, inplace = True)
fsol.to_csv('sol.csv', encoding = 'utf-8', index = False)
