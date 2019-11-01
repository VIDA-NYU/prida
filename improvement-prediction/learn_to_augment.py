#!/usr/bin/env python3
import json
import sys
from learning_task import *

if __name__ == '__main__':
    """This script reads a file with features generated by 
    generate_data_for_augmentation_learning_spark.py, and  
    generates a machine learning model to learn relative gains 
    in performance metrics
    """

    # Reads a parameter file in the format described in the README
    if len(sys.argv) == 1:
        params = json.load(open('params.json'))
    else:
        params = json.load(open(sys.argv[1]))

    # File with features and relative gains in performance after augmentation 
    augmentation_learning_data_filename = params['augmentation_learning_data_filename']

    # Number of folds for the cross-validation of the model
    n_splits = params['n_splits']

    # Reads data, learns models, and returns models and test_data for each data fold,
    # where the model can be a random forest or a linear regression, and test_data is a
    # list where each item has the index of the test instances and their true relative
    # performance gain, so we can adequately evaluate the models over them
    learning_task = LearningTask()
    learning_task.read_features_and_targets(augmentation_learning_data_filename)
    models, test_data = learning_task.execute_linear_regression(n_splits)
    
