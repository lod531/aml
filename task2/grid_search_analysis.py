from pprint import pprint
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import SVC
from normalized_weights import normalized_weights
from sklearn.metrics import balanced_accuracy_score, make_scorer



gs = pickle.load(open('svc.gridsearch', 'rb'))
print('best params', gs.best_params_)
print('all params', gs.cv_results_.keys())
print(gs.cv_results_['mean_test_score'])
