'''
Given a dataset filename and fractions X and Y, this script splits the dataset into training (with a fraction X of all tuples), 
validation (with a fraction Y of all tuples), and test (with a fraction 1 - X - Y of all tuples). The splitting guarantees that 
datasets present in one part of the data are not used in another.

This script additionally creates a column with binary classes based on a certain target variable. For now, the target variable and 
the threshold that are used to separate classes are modeled as constants.
'''

import sys
import pandas as pd
from sklearn.utils import shuffle

ALPHA = 0.7
TARGET = 'gain_in_r2_score'

def determine_classes_based_on_gain_in_r2_score(dataset):
  """This function determines the class of each row in the dataset based on the value 
  of column TARGET
  """
  gains = dataset[TARGET]
  classes = ['good_gain' if i > ALPHA else 'loss' for i in gains]
  dataset['classes'] = classes
  return dataset

def split_data(data, training_fraction, validation_fraction):
    
    keys = list(set([(row['query'], row['target']) for i, row in data.iterrows()])) 
    num_keys_training = int(training_fraction * len(keys)) 
    num_keys_validation = int(validation_fraction * len(keys)) 
    training_read = 0 
    validation_read = 0 
    training_rows = pd.DataFrame(columns=data.columns) 
    validation_rows = pd.DataFrame(columns=data.columns) 
    test_rows = pd.DataFrame(columns=data.columns) 
    for k in keys: 
        if training_read < num_keys_training: 
            training_rows = training_rows.append(data.loc[(data['query'] == k[0]) & (data['target'] == k[1])]) 
            training_read += 1 
        elif validation_read < num_keys_validation: 
            validation_rows = validation_rows.append(data.loc[(data['query'] == k[0]) & (data['target'] == k[1])]) 
            validation_read += 1 
        else: 
            test_rows = test_rows.append(data.loc[(data['query'] == k[0]) & (data['target'] == k[1])]) 
    return training_rows, validation_rows, test_rows 

if __name__ == '__main__':
    data = pd.read_csv(sys.argv[1])
    training_fraction = float(sys.argv[2])
    validation_fraction = float(sys.argv[3])
    training, validation, test = split_data(data, training_fraction, validation_fraction)

    training = determine_classes_based_on_gain_in_r2_score(training)
    ftraining = open('training-dataset-' + str(training_fraction * 100) + '-percent.csv', 'w')
    ftraining.write(training.to_csv(index=False))
    ftraining.close()

    validation = determine_classes_based_on_gain_in_r2_score(validation)
    fvalidation = open('validation-dataset-' + str(validation_fraction * 100) + '-percent.csv', 'w')
    fvalidation.write(validation.to_csv(index=False))
    fvalidation.close()

    test = determine_classes_based_on_gain_in_r2_score(test)
    ftest = open('test-dataset-' + str((1 - training_fraction - validation_fraction) * 100) + '-percent.csv', 'w')
    ftest.write(test.to_csv(index=False))
    ftest.close()
