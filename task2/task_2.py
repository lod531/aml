# ---------------------------------- TASK 2 ----------------------------------

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

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Grid search
C = [2 ** i for i in range(-6, 16)]
gamma = [2 ** i for i in range(-6, 16)]
kernel = ['linear', 'rbf']

param_grid = {'estimator__C': C, 'estimator__gamma': gamma, 'estimator__kernel':kernel}

classifier = GridSearchCV(OneVsOneClassifier(SVC()), param_grid = param_grid, cv = 2, verbose = 3, n_jobs = -1)
classifier.fit(X_train, y_train)
print("best params", classifier.best_params_)
'''
# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
from sklearn.metrics import balanced_accuracy_score, make_scorer
balanced_accuracy_scorer = make_scorer(balanced_accuracy_score, greater_is_better = True)
performances = cross_val_score(estimator = classifier, 
                               X = X_train, 
                               y = y_train, 
                               scoring = balanced_accuracy_scorer, 
                               cv = 5,
                               verbose = 2)
cv_score = performances.mean()
cv_variance = performances.std(r
'''

# Final solution
y_test_pred = classifier.predict(X_test)
sol = np.append(arr = ids_test.reshape(-1,1), values = y_test_pred.reshape(-1,1), axis = 1)
fsol = pd.DataFrame(sol)
fsol.rename(columns={0: 'id', 1: 'y'}, inplace = True)
fsol.to_csv('sol.csv', encoding = 'utf-8', index = False)
#print(cv_score)
