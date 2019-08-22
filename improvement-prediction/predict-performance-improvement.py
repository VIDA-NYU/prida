#!/usr/bin/env python3
import json
from augmentation_instance import *
from feature_factory import *
from constants import *

def parse_learning_data_filename(filename):
    with open(filename, 'r') as f:
        augmentation_instances = []
        for line in f:
            query_filename, candidate_filename, target, r2_score_before, r2_score_after = line.strip().split(SEPARATOR)
            fields = {}
            fields['query_filename'] = query_filename
            fields['candidate_filename'] = candidate_filename
            fields['target_name'] = target
            fields['initial_r2_score'] = float(r2_score_before)
            fields['final_r2_score'] = float(r2_score_after)
            augmentation_instances.append(AugmentationInstance(fields))
        return augmentation_instances

if __name__ == '__main__':

    params = json.load(open('params.json'))
    learning_data_filename = params['learning_data_filename']
    validation_n_splits = params['n_splits']
    output_filename = params['output_filename']
    augmentation_instances = parse_learning_data_filename(learning_data_filename)
    feature_factory = FeatureFactory()
    for instance in augmentation_instances:
        features = feature_factory.get_dataset_features(instance.get_query_dataset().get_data()) 
