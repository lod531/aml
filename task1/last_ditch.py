# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from xgboost import XGBRegressor
from sklearn.kernel_ridge import KernelRidge

# Importing the dataset
dataset_X_train = pd.read_csv('X_train.csv')
dataset_y_train = pd.read_csv('y_train.csv')
dataset_X_test = pd.read_csv('X_test.csv')
X_train = dataset_X_train.iloc[:,1:].values
y_train = dataset_y_train.iloc[:,1].values
ids_test = dataset_X_test.iloc[:,0].values
X_test = dataset_X_test.iloc[:,1:].values

# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'median', axis = 0)
X_train[:,:] = imputer.fit_transform(X_train[:,:])
X_test[:,:] = imputer.transform(X_test[:,:])

# Selecting features
from sklearn.ensemble import ExtraTreesClassifier, RandomForestRegressor
from sklearn.feature_selection import RFE 
clf = ExtraTreesClassifier(criterion = 'entropy', 
                           n_estimators = 1000,
                           max_features = 'auto', 
                           max_depth = 4,
                           min_samples_split = 3,
                           min_samples_leaf = 2,
                           min_weight_fraction_leaf = 0)
clf.fit(X_train, y_train)
rfe = RFE(estimator = clf, step = 2, verbose = 2)
X_train = rfe.fit_transform(X_train, y_train)
X_test = rfe.transform(X_test)

from sklearn.preprocessing import RobustScaler
rb_X = RobustScaler(with_centering = True, quantile_range = (23.0,74.0))
rb_y = RobustScaler(with_centering = False, quantile_range = (23.0,74.0))
X_train = rb_X.fit_transform(X_train)
X_test = rb_X.transform(X_test)
y_train = rb_y.fit_transform(y_train.reshape(-1, 1))[:,0]

# Applying Grid Search to find the best model and the best parameters
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import r2_score, make_scorer
r2_scorer = make_scorer(r2_score, greater_is_better = True)

# Fitting the SVR to the dataset
regressor = KernelRidge(alpha = 1.7e-2, kernel = 'laplacian', gamma = 1.44929e-4)
#regressor = RandomForestRegressor(n_estimators = 300, max_depth = 3)

# Final solution
regressor.fit(X_train, y_train)
y_test_pred = rb_y.inverse_transform(regressor.predict(X_test).reshape(-1,1))
y_test_pred = np.rint(y_test_pred)
sol = np.append(arr = ids_test.reshape(-1,1), values = y_test_pred.reshape(-1,1), axis = 1)
fsol = pd.DataFrame(sol)
fsol.rename(columns = {0: 'id', 1: 'y'}, inplace = True)
fsol.to_csv('last_ditch_sol.csv', encoding = 'utf-8', index = False)
print("DONE")
