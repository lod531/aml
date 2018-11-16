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

def get_heart_rate(beats=None, sampling_rate=300., smooth=False, size=3):
    # check inputs
    if beats is None:
        raise TypeError("Please specify the input beat indices.")

    if len(beats) < 2:
        raise ValueError("Not enough beats to compute heart rate.")

    # compute heart rate
    ts = beats[1:]
    hr = sampling_rate * (60. / np.diff(beats))

    # physiological limits
    #indx = np.nonzero(np.logical_and(hr >= 40, hr <= 200))
    #ts = ts[indx]
    #hr = hr[indx]

    return hr

def simple_sample_heartbeat_stats(arr):
    order = int(0.3 * SAMPLING_RATE)
    filtered, _, _ = biosppy.tools.filter_signal(signal=arr,
                                  ftype='FIR',
                                  band='bandpass',
                                  order=order,
                                  frequency=[3, 45],
                                  sampling_rate=SAMPLING_RATE)
    heart_beats = biosppy.signals.ecg.hamilton_segmenter(signal = filtered, sampling_rate = SAMPLING_RATE)['rpeaks']
    heart_rate_on_beats = get_heart_rate(beats = heart_beats, sampling_rate=SAMPLING_RATE)
    avg_heart_rate = np.mean(heart_rate_on_beats)
    median_heart_rate = np.median(heart_rate_on_beats)
    heart_rate_std = np.std(heart_rate_on_beats)
    return avg_heart_rate, median_heart_rate, heart_rate_std

def simple_dataset_heartbeat_stats(arr, labels):
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
        mean, median, std = simple_sample_heartbeat_stats(arr[i])
        results[label]['means'].append(mean)
        results[label]['medians'].append(median)
        results[label]['std_devs'].append(std)
    return results

def meta_dataset_heartbeat_stats(dict):
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

def mean_median_dev_percentiles_rpeak_qnadir_for_sample(arr):
    #assumes arr is a raw ecg signal to be filtered. 
    #This code could be very easily extended to make that optional
    SEARCH_WINDOW = 20
    order = int(0.3 * SAMPLING_RATE)
    data_point = X_train_data_clean[0]
    filtered, _, _ = biosppy.tools.filter_signal(signal=arr,
                                      ftype='FIR',
                                      band='bandpass',
                                      order=order,
                                      frequency=[3, 45],
                                      sampling_rate=SAMPLING_RATE)
    heart_beat_indexes = biosppy.signals.ecg.hamilton_segmenter(signal = filtered, 
                                                                sampling_rate = SAMPLING_RATE)['rpeaks']
    r_peaks = []
    q_nadirs = []
    for i in heart_beat_indexes:
        lower_bound = max(0, i-SEARCH_WINDOW)
        upper_bound = min(len(filtered), i+SEARCH_WINDOW)
        current_max = filtered[lower_bound]
        current_min = current_max
        for i in range(lower_bound, upper_bound):
            if filtered[i] > current_max:
                current_max = filtered[i]
            if filtered[i] < current_min:
                current_min = filtered[i]
        r_peaks.append(current_max)
        q_nadirs.append(current_min)
    result = {}
    result['rpeaks_percentiles'] = np.percentile(r_peaks, [i*10 for i in range(0, 11)])
    result['rpeaks_mean'] = np.mean(r_peaks)
    result['rpeaks_median'] = np.median(r_peaks)
    result['rpeaks_std_dev'] = np.std(r_peaks)
    result['qnadirs_percentiles'] = np.percentile(q_nadirs, [i*10 for i in range(0, 11)])
    result['qnadirs_mean'] = np.mean(q_nadirs)
    result['qnadirs_median'] = np.median(q_nadirs)
    result['qnadirs_std_dev'] = np.std(q_nadirs)
    return result
  
    
def mean_median_dev_percentiles_rpeak_qnadir_for_dataset(arr, labels):
    #The nesting of dictionaries here is fucking cancer, You don't have to tell me.
    r_peak_results = {}
    q_nadir_results = {}
    #Initialize results
    unique_labels = np.unique(labels)
    for i in range(0, len(unique_labels)):
        #For each label, I want a dict containing a list for all of my stats
        r_peak_results[unique_labels[i]] = {'means':[], 'medians':[], 'std_devs':[], 'percentiles':[]}
        q_nadir_results[unique_labels[i]] = {'means':[], 'medians':[], 'std_devs':[], 'percentiles':[]}
    for i in range(0, len(arr)):
        label = labels[i]
        sample_stats = mean_median_dev_rpeak_qnadir_for_sample(arr[i])
        r_peak_results[label]['means'].append(sample_stats['rpeaks_mean'])
        r_peak_results[label]['medians'].append(sample_stats['rpeaks_median'])
        r_peak_results[label]['std_devs'].append(sample_stats['rpeaks_std_dev'])
        r_peak_results[label]['percentiles'].append(sample_stats['rpeaks_percentiles'])
        q_nadir_results[label]['means'].append(sample_stats['qnadirs_mean'])
        q_nadir_results[label]['medians'].append(sample_stats['qnadirs_median'])
        q_nadir_results[label]['std_devs'].append(sample_stats['qnadirs_std_dev'])
        q_nadir_results[label]['percentiles'].append(sample_stats['qnadirs_percentiles'])
    return {'rpeak':r_peak_results, 'qnadir':q_nadir_results}

def meta_dataset_rpeak_qnadir_stats(result):
    #Pass the result of mean_median_dev_rpeak_qnadir_for_dataset into this bad boy
    print("Meta dataset stats for rpeak, qnadir")
    rpeak_results = result['rpeak']
    qnadir_results = result['qnadir']
    print(type(rpeak_results))
    print('R peak:')
    for label, stats in rpeak_results.items():
        print('Label' + str(label))
        mean_of_means = np.mean(stats['means'])
        mean_of_medians = np.mean(stats['medians'])
        mean_of_std_devs = np.mean(stats['std_devs'])
        print('Mean of means', mean_of_means)
        print('Mean of medians', mean_of_medians)
        print('Mean of standard deviations', mean_of_std_devs)
        
        std_dev_of_means = np.std(stats['means'])
        std_dev_of_medians = np.std(stats['medians'])
        std_dev_of_std_devs = np.std(stats['std_devs'])
        print('Standard deviation of means', std_dev_of_means)
        print('Standard deviation of medians', std_dev_of_medians)
        print('Standard deviation of standard deviations', std_dev_of_std_devs)
    print()
    print('Q nadir:')
    for label, stats in qnadir_results.items():
        print('Label' + str(label))
        mean_of_means = np.mean(stats['means'])
        mean_of_medians = np.mean(stats['medians'])
        mean_of_std_devs = np.mean(stats['std_devs'])
        print('Mean of means', mean_of_means)
        print('Mean of medians', mean_of_medians)
        print('Mean of standard deviations', mean_of_std_devs)
        
        std_dev_of_means = np.std(stats['means'])
        std_dev_of_medians = np.std(stats['medians'])
        std_dev_of_std_devs = np.std(stats['std_devs'])
        print('Standard deviation of means', std_dev_of_means)
        print('Standard deviation of medians', std_dev_of_medians)
        print('Standard deviation of standard deviations', std_dev_of_std_devs)

def ecg_features_for_sample(sample):
    mean_heart_rate, median_heart_rate, heart_rate_std = simple_sample_heartbeat_stats(sample)
    rpeak_qnadir = mean_median_dev_percentiles_rpeak_qnadir_for_sample(sample)
    rpeaks_mean = rpeak_qnadir['rpeaks_mean']
    rpeaks_median = rpeak_qnadir['rpeaks_median']
    rpeaks_std_dev = rpeak_qnadir['rpeaks_std_dev']
    rpeaks_percentiles = rpeak_qnadir['rpeaks_percentiles']
    print(rpeaks_percentiles)
    qnadirs_mean = rpeak_qnadir['qnadirs_mean']
    qnadirs_median = rpeak_qnadir['qnadirs_median'] 
    qnadirs_std_dev = rpeak_qnadir['qnadirs_std_dev']
    qnadirs_percentiles = rpeak_qnadir['qnadirs_percentiles']
    print(qnadirs_percentiles)
    feature_list = [mean_heart_rate, median_heart_rate, heart_rate_std, 
                       rpeaks_mean, rpeaks_median, rpeaks_std_dev,
                       qnadirs_mean, qnadirs_median, qnadirs_std_dev]
    for i in rpeaks_percentiles:
        feature_list.append(i)
    for i in qnadirs_percentiles:
        feature_list.append(i)
    return np.array(feature_list)

def ecg_features_for_dataset(dataset):
    result_list = []
    for i in range(0, len(dataset)):
        result_list.append(ecg_features_for_sample(dataset[i]))
    return np.array(result_list)
    

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
X_test_data_clean = clean_nan(X_test_data)
#4 rows appear to be corrupted or some shit, so.
#Their ids, relative to the unedited import, are
#2719, 3178, 4299, 4467, and they all belong to class 2
#Idk wtf.
#X_train_data_clean = np.delete(arr = X_train_data_clean, obj=[2719, 3178, 4299, 4467], axis=0)
#y_train_data = np.delete(arr = y_train_data, obj=[2719, 3178, 4299, 4467], axis=0)


summarized_training_dataset = ecg_features_for_dataset(X_train_data_clean)
pickle.dump(summarized_training_dataset, 
				open('mean_median_std_dev_heartrate_rpeaks_qnadirs_training.pickle', 'wb'))

summarized_testing_dataset = ecg_features_for_dataset(X_test_data_clean)
pickle.dump(summarized_testing_dataset,
				open('mean_median_std_dev_heartrate_rpeaks_qnadirs_testing.pickle', 'wb'))

print('Script executed in:', datetime.now()-script_start_time) 
