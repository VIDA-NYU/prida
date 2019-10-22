#!/usr/bin/env python3
import json
import sys
from recommender import *

if __name__ == '__main__':
    if len(sys.argv) == 1:
        params = json.load(open('params.json'))
    else:
        params = json.load(open(sys.argv[1]))

    recommender = Recommender()
    recommender.store_instances(params['learning_data_filename'])
    print('done storing instances')
    models = recommender.generate_models(params['augmentation_learning_data_filename'], params['n_splits'])
    # each model corresponds to the following tuple: rf, test_index, y_test
    for model in models:
        print(model['index_of_test_instances'])
