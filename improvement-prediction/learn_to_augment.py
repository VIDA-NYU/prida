#!/usr/bin/env python3
import json
import sys
from learning_task import *

if __name__ == '__main__':
    if len(sys.argv) == 1:
        params = json.load(open('params.json'))
    else:
        params = json.load(open(sys.argv[1]))
    augmentation_learning_data_filename = params['augmentation_learning_data_filename']
    n_splits = params['n_splits']
    output_filename = params['output_filename']
    learning_task = LearningTask()
    learning_task.read_data(augmentation_learning_data_filename)
    learning_task.execute_random_forest(n_splits) #, feature_ids=[4, 39, 8, 0, 3, 44, 16, 21, 34, 37, 7, 6, 29, 38, 40, 41, 43, 27])
