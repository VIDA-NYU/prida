#!/usr/bin/env python3
import json
import sys
from augmentation_instance import *
from feature_factory import *
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
        feature_factory_query = FeatureFactory(instance.get_joined_query_data())
        query_individual_metrics = feature_factory_query.get_individual_metrics(func=max_in_modulus)
        feature_factory_candidate = FeatureFactory(instance.get_joined_candidate_data())
        candidate_individual_metrics = feature_factory_candidate.get_individual_metrics(func=max_in_modulus)
        feature_factory_full_dataset = FeatureFactory(instance.get_joined_data())
        full_dataset_pairwise_metrics = feature_factory_full_dataset.get_pairwise_metrics(func=max_in_modulus)
        pairwise_metrics_with_target = feature_factory_full_dataset.get_pairwise_metrics_with_target(instance.get_target_column_name(),
                                                                                                     func=max_in_modulus)
        query_pairwise_metrics_with_target = feature_factory_query.get_pairwise_metrics_with_target(instance.get_target_column_name(),
                                                                                                    func=max_in_modulus)
        feature_factory_candidate_with_target = FeatureFactory(instance.get_joined_candidate_data_and_target())
        candidate_pairwise_metrics_with_target = feature_factory_candidate_with_target.get_pairwise_metrics_with_target(instance.get_target_column_name(),
                                                                                                                        func=max_in_modulus)
        max_in_modulus_pearson_difference = feature_factory_candidate_with_target.compute_difference_in_pearsons_wrt_target(feature_factory_query.get_max_pearson_wrt_target(instance.get_target_column_name()),
                                                                                                                            instance.get_target_column_name())
        
        r2_gain = instance.compute_r2_gain()

        learning_features = query_individual_metrics + \
                            candidate_individual_metrics + \
                            full_dataset_pairwise_metrics + \
                            pairwise_metrics_with_target + \
                            query_pairwise_metrics_with_target + \
                            candidate_pairwise_metrics_with_target + \
                            [max_in_modulus_pearson_difference]
        learning_target = r2_gain
        learning_task.add_learning_instance(learning_features, learning_target)
        i += 1
        if (i % 100 == 0):
            print(i)
    learning_task.dump_learning_instances(augmentation_learning_data_filename)
    print('done processing augmentation instances and creating data')
