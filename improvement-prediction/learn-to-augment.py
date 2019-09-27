#!/usr/bin/env python3
import json
from learning_task import *

if __name__ == '__main__':

    params = json.load(open('params.json'))
    augmentation_learning_data_filename = params['augmentation_learning_data_filename']
    n_splits = params['n_splits']
    output_filename = params['output_filename']
    learning_task = LearningTask()
    learning_task.read_data(augmentation_learning_data_filename)
    learning_task.execute_linear_regression(n_splits)
