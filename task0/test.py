import csv
import pandas
from sklearn.metrics import mean_squared_error
from sklearn import linear_model

'''
TRAINING_PERCENTAGE = 0.7



# SPLITTING THE DATA ################################################################
complete_set = pandas.read_csv('train.csv')

number_of_training_examples = int(complete_set.shape[0]*TRAINING_PERCENTAGE)
training_set = complete_set[0:number_of_training_examples]
validation_set = complete_set[number_of_training_examples:complete_set.shape[0]]

training_data = training_set.drop(['Id','y'], axis=1)
training_labels = training_set['y']


validation_data = validation_set.drop(['Id', 'y'], axis=1)
validation_labels = validation_set['y']

# SPLITTING THE DATA ################################################################


regr = linear_model.LinearRegression()
regr.fit(training_data, training_labels)

predicted_labels = regr.predict(validation_data)

RMSE = mean_squared_error(validation_labels, predicted_labels)**0.5
print(RMSE)

'''

training_set = pandas.read_csv('train.csv')
training_data = training_set.drop(['Id','y'], axis=1)
training_labels = training_set['y']


regr = linear_model.LinearRegression()
regr.fit(training_data, training_labels)

test_set = pandas.read_csv('test.csv')
print(test_set['Id'])
test_data = test_set.drop('Id', axis=1)
predictions = regr.predict(test_data)

predictions_data_frame = pandas.DataFrame({'y':predictions})
final_frame = pandas.DataFrame(test_set['Id']).join(predictions_data_frame)


final_frame.to_csv('output_test.csv', index=False)
print(final_frame.shape)
