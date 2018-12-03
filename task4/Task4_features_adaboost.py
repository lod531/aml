# ---------------------------------- TASK 4 ----------------------------------

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn

# Importing the dataset
dataset_X_train = pd.read_csv('X_train_custom.csv')
dataset_y_train = pd.read_csv('train_target.csv')
X_train = dataset_X_train.iloc[:,1:].values
y_train = dataset_y_train.iloc[:,1].values

dataset_X_test = pd.read_csv('X_test_custom.csv')
ids_test = dataset_X_test.iloc[:,0].values
X_test = dataset_X_test.iloc[:,1:].values

# =============================================================================
# # Missing values
# 
# from sklearn.preprocessing import Imputer
# imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
# 
# imputer = imputer.fit(X_train[:, :])
# X_train[:, :] = imputer.transform(X_train[:,:])
# 
# # Missing values on test set
# X_test[:,:] = imputer.transform(X_test[:,:])
# =============================================================================


# Selecting features
# =============================================================================
# from sklearn.ensemble import ExtraTreesClassifier
# from sklearn.feature_selection import SelectFromModel
# clf = ExtraTreesClassifier(criterion = 'entropy', 
#                            n_estimators = 440,
#                            max_features = 'auto', 
#                            max_depth = 3,
#                            min_samples_split = 2,
#                            min_samples_leaf = 1,
#                            min_weight_fraction_leaf = 0)
# =============================================================================

# Building the classifier (XGB)
from xgboost import XGBClassifier
clf = XGBClassifier(max_depth=4, 
                             learning_rate=0.1, 
                             n_estimators=240, 
                             objective='reg:linear', 
                             gamma=0.967, 
                             min_child_weight=1, 
                             max_delta_step=0, 
                             subsample=1, 
                             colsample_bytree=1, 
                             colsample_bylevel=1, 
                             reg_alpha=1.68e0, 
                             reg_lambda=1, 
                             scale_pos_weight=1, 
                             base_score=0.5, 
                             missing=None)


clf = clf.fit(X_train, y_train)

# =============================================================================
# model = SelectFromModel(clf, prefit=True)
# X_train = model.transform(X_train)
# X_test = model.transform(X_test)
# =============================================================================

#Building classifier ADABoost
from sklearn.ensemble import AdaBoostClassifier
classifier = AdaBoostClassifier(clf,
                         algorithm="SAMME",
                         n_estimators=100)


# =============================================================================
# # SVC
# from sklearn import svm
# classifier = svm.SVC(C=1.3, cache_size=200, class_weight=None, coef0=0.0,
#     decision_function_shape='ovo', degree=3, gamma='scale', kernel='rbf',
#     max_iter=-1, probability=True, shrinking=True,
#     tol=0.001, verbose=False)
# # cv score 0.67 with svc
# =============================================================================


# Cross Validation
from sklearn.model_selection import RepeatedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif, f_regression
rkf = RepeatedKFold(n_splits=2, n_repeats=20)
anova_filter = SelectKBest(f_classif, k=400)
X_train = anova_filter.fit_transform(X_train, y_train)
scores = cross_val_score(classifier, X_train, y_train, cv=rkf,scoring = 'roc_auc',
                         verbose=3)
print(scores.mean())
print(scores.var())

# Final solution
X_test = anova_filter.transform(X_test)

classifier.fit(X_train, y_train)
y_test_pred = classifier.predict_proba(X_test)[:,1]
sol = np.append(arr = ids_test.reshape(-1,1), values = y_test_pred.reshape(-1,1), axis = 1)
fsol = pd.DataFrame(sol)
fsol.rename(columns={0: 'id', 1: 'y'}, inplace = True)
fsol.to_csv('sol.csv', encoding = 'utf-8', index = False)
