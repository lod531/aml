import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import SVC
from normalized_weights import normalized_weights
from sklearn.metrics import balanced_accuracy_score, make_scorer


## Importing the dataset
#dataset_X_train = pd.read_csv('X_train.csv')
#dataset_y_train = pd.read_csv('y_train.csv')
#X_train = dataset_X_train.iloc[:,1:].values
#y_train = dataset_y_train.iloc[:,1].values
#
#dataset_X_test = pd.read_csv('X_test.csv')
#ids_test = dataset_X_test.iloc[:,0].values
#X_test = dataset_X_test.iloc[:,1:].values

dataset_X_train = pd.read_csv('X_train.csv')
dataset_y_train = pd.read_csv('y_train.csv')
X_train = dataset_X_train.iloc[:,1:].values
y_train = dataset_y_train.iloc[:,1].values

y_indexes = [i for i, x in enumerate(y_train) if x == 1.0]
#make it predictable
np.random.seed(42)
#samples to be omitted
random_choices = np.random.choice(y_indexes, 2700, replace=False)
random_choices = y_indexes[2700:]
X_train = np.delete(X_train, random_choices, axis = 0)
y_train = np.delete(y_train, random_choices, axis = 0)

dataset_X_test = pd.read_csv('X_test.csv')
ids_test = dataset_X_test.iloc[:,0].values
X_test = dataset_X_test.iloc[:,1:].values


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Grid search
#C = [2 ** i for i in range(-6, 16)]
#gamma = [2 ** i for i in range(-6, 16)]
#kernel = ['linear', 'rbf']
#
#C = [2 ** (2*i) for i in range(-3, 8)]
#gamma = [2 ** (2*i) for i in range(-3, 8)]
#kernel = ['linear', 'rbf']

#C = [0.662]
#gamma = [0.000967]
#Best params: {'gamma': 0.0001, 'C': 0.0008, 'kernel': 'linear'}
#Best score: 0.7005555
gamma = [0.000967]
C = [0.0008]
C = [1]
#C = [0.662]
#gamma = [0.0009679]
#gamma = [0.0004]
kernel = ['rbf', 'linear']


param_grid = {'C': C, 'gamma': gamma, 'kernel':kernel}
classifier = GridSearchCV(SVC(class_weight = 'balanced', decision_function_shape = 'ovo', random_state = 0), param_grid = param_grid, cv = 8, verbose = 3, n_jobs = -1, scoring = make_scorer(balanced_accuracy_score))
#classifier = SVC(C = 1.0,
#                                    kernel = 'rbf', 
#                                    random_state = 0,
#                                    class_weight = 'balanced',
#                                    gamma = 0.967e-3,
#                                    decision_function_shape = 'ovo')
##


classifier.fit(X_train,y_train)
pickle.dump(classifier, open('svc.gridsearch', 'wb'))
print("Best score:", classifier.best_score_)
print("Best params:", classifier.best_params_)



# Final solution
y_test_pred = classifier.predict(X_test)
sol = np.append(arr = ids_test.reshape(-1,1), values = y_test_pred.reshape(-1,1), axis = 1)
fsol = pd.DataFrame(sol)
fsol.rename(columns={0: 'id', 1: 'y'}, inplace = True)
fsol.to_csv('sol.csv', encoding = 'utf-8', index = False)
#print(cv_score)
