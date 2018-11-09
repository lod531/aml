# Importing the libraries
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.externals.joblib import parallel_backend
import xgboost as xgb
from pprint import pprint

# Importing the dataset
dataset_X = pd.read_csv('X_train.csv')
dataset_y = pd.read_csv('y_train.csv')
X_train = dataset_X.iloc[:,1:].values
y_train = dataset_y.iloc[:,1].values

testset = pd.read_csv('X_test.csv')
X_test = testset.iloc[:,1:].values
ids_test = testset.iloc[:,0].values

# Missing values
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer.fit(np.concatenate((X_train[:,:],X_test[:,:])))
X_train[:,:] = imputer.transform(X_train[:,:])
X_test[:,:] = imputer.transform(X_test[:,:])
X_train_new = X_train

grid_search = pickle.load(open('andy_search.tmp', 'rb'))   

best_score = grid_search.best_score_
best_parameters = grid_search.best_params_
print(best_score)
print(best_parameters)

# Predicting the new results
regressor = grid_search
y_pred = regressor.predict(X_test)

# Rounding to the nearest integer
#y_pred = np.rint(y_pred)

'''
# Final Solution
sol = np.append(arr = ids_test.reshape(-1,1), values = y_pred.reshape(-1,1), axis = 1)
fsol = pd.DataFrame(sol)
fsol.rename(columns={0: 'id', 1: 'y'}, inplace = True)
fsol.to_csv('sol.csv', encoding = 'utf-8', index = False)
'''
