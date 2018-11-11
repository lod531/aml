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

X_test_ids = pickle.load(open('X_test_ids.pickle', 'rb'))
X_test_data = pickle.load(open('mean_median_std_dev_heartrate_rpeaks_qnadirs_testing.pickle', 'rb'))


clf = pickle.load(open('grid_search_results.pickle', 'rb'))
print(clf.best_params_)
print(clf.best_score_)

y_test_pred = clf.predict(X_test_data)
sol = np.append(arr = X_test_ids.reshape(-1,1), values = y_test_pred.reshape(-1,1), axis = 1)
fsol = pd.DataFrame(sol)
fsol.rename(columns={0: 'id', 1: 'y'}, inplace = True)
fsol.to_csv('sol.csv', encoding = 'utf-8', index = False)

print('Script executed in:', datetime.now()-script_start_time) 
