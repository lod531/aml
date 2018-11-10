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

SAMPLING_RATE = 300 #Hz

#just some simple counting, to know how much and of what We have
def simple_data_stats(arr, name):
    print('SIMPLE DATA STATS FOR ' + name)
    row_lengths = []
    for i in range(0, len(arr)):
        current_row_length = 0
        for j in range(0, len(arr[i])):
            if not np.isnan(arr[i][j]):
                current_row_length += 1
        row_lengths.append(current_row_length)

    print('min length:', min(row_lengths), 'max length:', max(row_lengths))
    print('index of min_length', row_lengths.index(min(row_lengths)))
    print('index of max_length', row_lengths.index(max(row_lengths)))
    print('mean:', np.mean(row_lengths))
    print('standard deviation', np.std(row_lengths))
    print('standard deviation', np.std(row_lengths))

def simple_label_stats(arr, name):
    print('SIMPLE LABEL STATS FOR ' + name)
    unique_values, unique_counts = np.unique(arr, return_counts = True)
    print('number of labels (expected 4)', len(unique_values))
    unique_ratios = [i/len(arr) for i in unique_counts]
    for i in range(0, len(unique_values)):
        print('% of dataset that is labelled with ' + str(unique_values[i]), unique_ratios[i])

def clean_nan(arr):
    #assumes 2D
    result = []
    for i in range(0, len(arr)):
        clean_current_column = []
        no_nans_encountered = True
        #nans are at the end of entries (I checked), so once a nan has been encountered
        #the rest of the row can be ignored
        j = 0
        while no_nans_encountered and j < len(arr[i]):
            if not np.isnan(arr[i][j]):
                clean_current_column.append(arr[i][j])
                if not no_nans_encountered:
                    print('Well, fuck, there\'s nans after values')
            else:
                no_nans_encountered = False
            j += 1
        result.append(clean_current_column)
    return result

def simple_sample_stats(arr):
    stats = biosppy.signals.ecg.ecg(arr, sampling_rate=SAMPLING_RATE, show=False)
    heart_rate_by_second = stats['heart_rate']
    avg_heart_rate = np.mean(heart_rate_by_second)
    median_heart_rate = np.median(heart_rate_by_second)
    heart_rate_std = np.std(heart_rate_by_second)
    return avg_heart_rate, median_heart_rate, heart_rate_std

def simple_dataset_stats(arr, labels):
    #Arr must be clean, i.e. must not contain NaN values.
    #use clean_nan to clean a dataset
    #I would like simple sample stats for each sample, grouped by 
    #label
    #For each label I would like a list of means, medians and stddevs
    results = {}
    #Initialize results
    unique_labels = np.unique(labels)
    for i in range(0, len(unique_labels)):
        #for each label, I want a list of lists of length 3
        results[unique_labels[i]] = {'means':[], 'medians':[], 'std_devs':[]}
    for i in range(0, len(arr)):
        label = labels[i]
        mean, median, std = simple_sample_stats(arr[i])
        if np.isnan(mean) or np.isnan(median) or np.isnan(std):
            print('NaN detected! Row id: ', i)
        results[label]['means'].append(mean)
        results[label]['medians'].append(median)
        results[label]['std_devs'].append(std)
    return results

def meta_dataset_stats(dict):
    for label, stats in dict.items():
        print('STATS FOR LABEL ' + str(label))
        mean_of_means = np.mean(stats['means'])
        mean_of_medians = np.mean(stats['medians'])
        mean_of_std_devs = np.mean(stats['std_devs'])
        print('Mean of means:', mean_of_means)
        print('Mean of medians:', mean_of_medians)
        print('mean of standard deviations:', mean_of_std_devs)

        dev_of_means = np.std(stats['means'])
        dev_of_medians = np.std(stats['medians'])
        dev_of_std_devs = np.std(stats['std_devs'])
        print('Standard deviation of means', dev_of_means)
        print('Standard deviation of medians', dev_of_medians)
        print('Standard deviation of std_devs', dev_of_std_devs)



#time the script
script_start_time = datetime.now()



# Make it predictable
np.random.seed(42)

# Importing the dataset
dataset_X_train = pd.read_csv('X_train.csv')
dataset_y_train = pd.read_csv('y_train.csv')
X_train_data = dataset_X_train.iloc[:,1:].values
y_train_data = dataset_y_train.iloc[:,1].values

dataset_X_test = pd.read_csv('X_test.csv', nrows = 1)
X_test_ids = dataset_X_test.iloc[:,0].values
X_test_data = dataset_X_test.iloc[:,1:].values

#Unfortunately, NaN values are present, which won't do for biosppy analysis.
#So let's get a version going that doesn't feature them.
X_train_data_clean = clean_nan(X_train_data)


#simple_data_stats(X_train_data, 'train data')
#simple_data_stats(X_test_data, 'test data')
#
#simple_label_stats(y_train_data, 'train data')

#calculate mean, median and stddev for each sample
#then look at the stddev from the mean of the samples for each ecg



#signal = test
#sampling_rate = 300
#length = len(signal)
#T = (length - 1) / sampling_rate
#tsa = np.linspace(0, T, length, endpoint=False)
##biosppy.plotting.plot_ecg(raw = test, ts = tsa)
#testt = biosppy.signals.ecg.ecg(signal = test, sampling_rate = 300, show=False)
#print('LENGTH0', len(X_train_data))
#print('LENGTH1', len(X_train_data_clean))
#print(simple_sample_stats(X_train_data_clean[1]))
#pprint(simple_dataset_stats(X_train_data_clean, y_train_data))

#test = simple_dataset_stats(X_train_data_clean, y_train_data)
#meta_dataset_stats(test)
troublesome_ids = [2719, 3178, 4299, 4467]
for idd in troublesome_ids:
    print('id ' + str(idd), 'label', y_train_data[idd], 'length', len(X_train_data[idd]))
    biosppy.signals.ecg.ecg(signal = X_train_data[2719], sampling_rate = SAMPLING_RATE, show=True)



# testing set
    

## Feature Scaling
#from sklearn.preprocessing import StandardScaler
#sc = StandardScaler()
#X_train = sc.fit_transform(X_train)
#X_test = sc.transform(X_test)
#
## Grid search
##C = [2 ** i for i in range(-6, 16)]
##gamma = [2 ** i for i in range(-6, 16)]
##kernel = ['linear', 'rbf']
##
##C = [2 ** (2*i) for i in range(-3, 8)]
##gamma = [2 ** (2*i) for i in range(-3, 8)]
##kernel = ['linear', 'rbf']
#
##C = [0.662]
##gamma = [0.000967]
##Best params: {'gamma': 0.0001, 'C': 0.0008, 'kernel': 'linear'}
##Best score: 0.7005555
#gamma = [0.000967]
#C = [0.0008]
#C = [1]
##C = [0.662]
##gamma = [0.0009679]
##gamma = [0.0004]
#kernel = ['rbf', 'linear']
#
#
#param_grid = {'C': C, 'gamma': gamma, 'kernel':kernel}
#classifier = GridSearchCV(SVC(class_weight = 'balanced', decision_function_shape = 'ovo', random_state = 0), param_grid = param_grid, cv = 8, verbose = 3, n_jobs = -1, scoring = make_scorer(balanced_accuracy_score))
##classifier = SVC(C = 1.0,
##                                    kernel = 'rbf', 
##                                    random_state = 0,
##                                    class_weight = 'balanced',
##                                    gamma = 0.967e-3,
##                                    decision_function_shape = 'ovo')
###
#
#
#classifier.fit(X_train,y_train)
#pickle.dump(classifier, open('svc.gridsearch', 'wb'))
#print("Best score:", classifier.best_score_)
#print("Best params:", classifier.best_params_)
#
#
#
## Final solution
#y_test_pred = classifier.predict(X_test)
#sol = np.append(arr = ids_test.reshape(-1,1), values = y_test_pred.reshape(-1,1), axis = 1)
#fsol = pd.DataFrame(sol)
#fsol.rename(columns={0: 'id', 1: 'y'}, inplace = True)
#fsol.to_csv('sol.csv', encoding = 'utf-8', index = False)
##print(cv_score)


print('Script executed in:', datetime.now()-script_start_time)
