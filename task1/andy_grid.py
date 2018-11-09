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

'''
# Features selection
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
clf = ExtraTreesClassifier(n_estimators=300)
clf = clf.fit(X_train, y_train)
model = SelectFromModel(clf, prefit = True)
X_train_new = model.transform(X_train)
'''


'''
regressor_xgb = xgb.XGBRegressor(n_estimators = 400, n_jobs = -1)
from sklearn.feature_selection import RFE
rfe = RFE(estimator = regressor_xgb, step = 400, verbose = 2)
X_train = rfe.fit_transform(X_train, y_train)
X_test = rfe.transform(X_test)
X_train_new = X_test
print(X_train_new.shape)
print(X_test.shape)
'''




from sklearn.metrics import r2_score, make_scorer
r2_scorer = make_scorer(r2_score, greater_is_better = True)

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 300, stop = 1000, num = 7)]
# Learninr rate
learning_rate = [x/500.0 for x in range(2, 11)]
# Max tree depth
max_depth = [4, 6, 8, 10]
# Row sampling
subsample = [0.5, 0.75, 1]
# Column sampling
colsample_bytree = [0.4, 0.6, 0.8, 1]




random_grid = {'n_estimators': n_estimators,
               'learning_rate': learning_rate,
               'max_depth': max_depth,
               'subsample': subsample,
               'colsample_bytree': colsample_bytree}

pprint(random_grid)

grid_search = GridSearchCV(estimator = xgb.XGBRegressor(nthread = 1),
                           param_grid = random_grid,
                           scoring = r2_scorer,
                           cv = 3,
                           verbose = 2,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train_new, y_train)
pickle.dump(grid_search, open('andy_search.tmp', 'wb'))   

best_score = grid_search.best_score_
best_parameters = grid_search.best_params_

# Predicting the new results
regressor = grid_search
y_pred = regressor.predict(X_test_new)

# Rounding to the nearest integer
#y_pred = np.rint(y_pred)

# Final Solution
sol = np.append(arr = ids_test.reshape(-1,1), values = y_pred.reshape(-1,1), axis = 1)
fsol = pd.DataFrame(sol)
fsol.rename(columns={0: 'id', 1: 'y'}, inplace = True)
fsol.to_csv('sol.csv', encoding = 'utf-8', index = False)
