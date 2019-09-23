#!/usr/bin/env python3
import json
from augmentation_instance import *
from feature_factory import *
from constants import *

def parse_learning_data_filename(filename):
    with open(filename, 'r') as f:
        augmentation_instances = []
        for line in f:
            query_filename, target, candidate_filename, r2_score_before, r2_score_after = line.strip().split(SEPARATOR)
            fields = {}
            fields['query_filename'] = query_filename
            fields['target_name'] = target
            fields['candidate_filename'] = candidate_filename
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
    for instance in augmentation_instances:
        feature_factory_query = FeatureFactory(instance.get_joined_query_data())
        query_individual_metrics = feature_factory_query.get_individual_metrics()
        feature_factory_candidate = FeatureFactory(instance.get_joined_candidate_data())
        candidate_individual_metrics = feature_factory_candidate.get_individual_metrics()
        feature_factory_full_dataset = FeatureFactory(instance.get_joined_data())
        full_dataset_pairwise_metrics = feature_factory_full_dataset.get_pairwise_metrics()
        pairwise_metrics_with_target = feature_factory_full_dataset.get_pairwise_metrics_with_target(instance.get_target_column_name())
        print(len(query_individual_metrics), len(candidate_individual_metrics), len(full_dataset_pairwise_metrics), len(pairwise_metrics_with_target))
        #TODO (1) use func == max(np.fabs) (2) formalize the gain with r2_score attributes (3) model the regression task
