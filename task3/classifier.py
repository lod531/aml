import pickle
from datetime import datetime
from pprint import pprint
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score, make_scorer
import biosppy
import matplotlib
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score

#time the script
script_start_time = datetime.now()
# Make it predictable
np.random.seed(42)

# Importing the dataset
#dataset_X_train = pd.read_csv('X_train.csv')
dataset_y_train = pd.read_csv('y_train.csv')
X_train_data = pickle.load(open('mean_median_std_dev_heartrate_rpeaks_qnadirs_training.pickle', 'rb'))
y_train_data = pickle.load(open('y_train_data.pickle', 'rb'))

dataset_X_test = pd.read_csv('X_test.csv', nrows = 1)
#X_test_ids = dataset_X_test.iloc[:,0].values
X_test_ids = pickle.load(open('X_test_ids.pickle', 'rb'))
X_test_data = pickle.load(open('mean_median_std_dev_heartrate_rpeaks_qnadirs_testing.pickle', 'rb'))


parameters = {
    #"loss":["deviance"],
    "learning_rate": [0.01],
    "min_samples_split": [20],
    'min_samples_split': [19],
    "min_samples_leaf": [2],
    "max_depth": [8, 9, 10],
    "max_features": ["log2","sqrt"],
    "criterion": ["friedman_mse"],
    #"subsample":[0.5, 0.618, 0.8, 0.85, 0.9, 0.95, 1.0],
    "subsample":np.linspace(0.3, 0.5, 10),
    "n_estimators":[130]
    }

clf = GridSearchCV(GradientBoostingClassifier(), parameters, cv=5, n_jobs=-1, scoring='f1_micro',
                        verbose = 10)
clf.fit(X_train_data, y_train_data)
pickle.dump(clf, open('grid_search_results.pickle', 'wb'))
print(clf.best_params_)
print(clf.best_score_)

y_test_pred = clf.predict(X_test_data)
sol = np.append(arr = X_test_ids.reshape(-1,1), values = y_test_pred.reshape(-1,1), axis = 1)
fsol = pd.DataFrame(sol)
fsol.rename(columns={0: 'id', 1: 'y'}, inplace = True)
fsol.to_csv('sol.csv', encoding = 'utf-8', index = False)

print('Script executed in:', datetime.now()-script_start_time) 
