#!/usr/bin/env python3
import json
import sys
from augmentation_instance import *
from util.file_parser import *
from util.file import *
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
    learning_features = []
    learning_targets = []
    
    i = 0
    for instance in augmentation_instances:
        learning_features.append(instance.generate_features())
        learning_targets.append(instance.compute_gain_in_r2_score())
        i += 1
        if (i % 100 == 0):
            print(i)
    dump_learning_instances(augmentation_learning_data_filename, learning_features, learning_targets)
    print('done processing augmentation instances and creating data')
