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
    models, test_data = recommender.generate_models_and_test_data(params['augmentation_learning_data_filename'], params['n_splits'])
    print('done creating models and corresponding test_data')
    for model, data in zip(models, test_data):
        recommender.predict_gains_for_candidate_datasets(model, data)
        break
