# Importing the libraries
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.externals.joblib import parallel_backend

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

# Features selection
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
clf = ExtraTreesClassifier(n_estimators=300)
clf = clf.fit(X_train, y_train)
model = SelectFromModel(clf, prefit = True)
X_train_new = model.transform(X_train)

# Selecting features on test set
X_test_new = model.transform(X_test)

# Feature scaling
'''from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X_train_new = sc_X.fit_transform(X_train_new)
X_test_new = sc_X.transform(X_test_new)
y_train = sc_y.fit_transform(y_train.reshape(-1, 1))[:,0]'''

# Applying Grid Search to find the best model and the best parameters
from sklearn.ensemble import RandomForestRegressor
'''from sklearn.model_selection import GridSearchCV'''
from sklearn.metrics import r2_score, make_scorer
r2_scorer = make_scorer(r2_score, greater_is_better = True)

'''
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
'''
    # Number of trees in random forest REDUCED UPPER BOUND TO 2
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 2)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree REDUCED TO 2 
max_depth = [int(x) for x in np.linspace(10, 110, num = 2)]
max_depth.append(None)
# Minimum number of samples required to split a node DELETED 10
min_samples_split = [2, 5]
# Minimum number of samples required at each leaf node DELETED 4
min_samples_leaf = [1, 2]
# Method of selecting samples for training each tree
bootstrap = [True, False]


'''
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
'''
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features}

print('STARTING GRID SEARCH')
grid_search = GridSearchCV(estimator = RandomForestRegressor(),
                           param_grid = random_grid,
                           scoring = r2_scorer,
                           cv = 2,
                           verbose = 3,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train_new, y_train)
pickle.dump(grid_search, open('small_grid_search.tmp', 'wb'))   
print('END GRID SEARCH')
best_score = grid_search.best_score_
best_parameters = grid_search.best_params_



regressor = grid_search
# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
print("Beginning of the end")
performances = cross_val_score(estimator = regressor, 
                             X = X_train_new, 
                             y = y_train, 
                             scoring = r2_scorer, 
                             cv = 4)
performances.mean()
performances.std()

# Predicting the new results
y_pred = regressor.predict(X_test_new)

# Rounding to the nearest integer
y_pred = np.rint(y_pred)

# Final Solution
sol = np.append(arr = ids_test.reshape(-1,1), values = y_pred.reshape(-1,1), axis = 1)
fsol = pd.DataFrame(sol)
fsol.rename(columns={0: 'id', 1: 'y'}, inplace = True)
fsol.to_csv('sol.csv', encoding = 'utf-8', index = False)
