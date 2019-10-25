#!/usr/bin/env python3
import json
import sys
from augmentation_instance import *
from util.file_parser import *
from learning_task import *

if __name__ == '__main__':

    if len(sys.argv) == 1:
        params = json.load(open('params.json'))
    else:
        params = json.load(open(sys.argv[1]))
    learning_data_filename = params['learning_data_filename']
    augmentation_learning_data_filename = params['augmentation_learning_data_filename']
    augmentation_instances = parse_augmentation_instances(learning_data_filename)
    print('Done parsing instances')
    learning_task = LearningTask()
    i = 0
    for instance in augmentation_instances:
        learning_task.add_learning_instance(instance.generate_features(), instance.compute_gain_in_r2_score())
        i += 1
        if (i % 100 == 0):
            print(i)
    learning_task.dump_learning_instances(augmentation_learning_data_filename)
    print('done processing augmentation instances and creating data')
