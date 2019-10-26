import numpy as np
from dataset import *
from feature_factory import *
from util.metrics import *

class AugmentationInstance:
    def __init__(self, instance_values, use_hdfs=False, hdfs_address=None, hdfs_user=None):
        self.query_filename = instance_values['query_filename']
        self.query_dataset = Dataset()
        self.query_dataset.initialize_from_filename(self.query_filename, use_hdfs, hdfs_address, hdfs_user)
        self.candidate_filename = instance_values['candidate_filename']
        self.candidate_dataset = Dataset()
        self.candidate_dataset.initialize_from_filename(self.candidate_filename, use_hdfs, hdfs_address, hdfs_user)
        self.target_name = instance_values['target_name']

        if len(instance_values.keys()) == 5:
            self.r2_score_before = instance_values['r2_score_before']
            self.r2_score_after = instance_values['r2_score_after']

        # for test instances, we do not have r2 values
        elif len(instance_values.keys()) == 3:
            self.r2_score_before = np.nan
            self.r2_score_after = np.nan
        self.joined_dataset = self.join_query_and_candidate_datasets()
        
    def get_query_dataset(self):
        return self.query_dataset

    def get_candidate_dataset(self):
        return self.candidate_dataset

    def get_query_filename(self):
        return self.query_filename

    def get_candidate_filename(self):
        return self.candidate_filename

    def join_query_and_candidate_datasets(self):
        result_data = self.query_dataset.join_with(self.candidate_dataset, key='key-for-ranking')
        dataset = Dataset()
        dataset.initialize_from_data_and_column_names(result_data, result_data.columns)
        return dataset

    def get_joined_query_data(self):
        query_column_names = self.query_dataset.get_column_names()
        return self.joined_dataset.get_data_columns(query_column_names, '_left')

    def get_joined_candidate_data(self):
        candidate_column_names = self.candidate_dataset.get_column_names()
        return self.joined_dataset.get_data_columns(candidate_column_names, '_right')

    def get_joined_candidate_data_and_target(self):
        column_names = self.candidate_dataset.get_column_names().tolist() + [self.target_name] 
        return self.joined_dataset.get_data_columns(column_names, '_right')

    def get_joined_data(self):
        return self.joined_dataset.get_data()

    def get_target_column_name(self):
        return self.target_name

    def compute_gain_in_r2_score(self):
        return compute_r2_gain(self.r2_score_before, self.r2_score_after)

    def compute_pairwise_metrics(self):
        feature_factory_full_dataset = FeatureFactory(self.get_joined_data())
        fd_metrics = feature_factory_full_dataset.get_pairwise_metrics(func=max_in_modulus)
        metrics_with_target = feature_factory_full_dataset.get_pairwise_metrics_with_target(self.target_name,
                                                                                            func=max_in_modulus)
        feature_factory_candidate_with_target = FeatureFactory(self.get_joined_candidate_data_and_target())
        candidate_metrics_with_target = feature_factory_candidate_with_target.get_pairwise_metrics_with_target(self.target_name,
                                                                                                               func=max_in_modulus)
        #FIXME avoid generating feature_factory_query twice 
        feature_factory_query = FeatureFactory(self.get_joined_query_data())
        query_metrics_with_target = feature_factory_query.get_pairwise_metrics_with_target(self.target_name,
                                                                                                    func=max_in_modulus)

        pearson_difference_wrt_target = feature_factory_candidate_with_target.compute_difference_in_pearsons_wrt_target(
            feature_factory_query.get_max_pearson_wrt_target(self.target_name), self.target_name)

        difference_in_numbers_of_rows = feature_factory_candidate_with_target.compute_percentual_difference_in_number_of_rows(self.query_dataset.get_data().shape[0])
        return fd_metrics + metrics_with_target + query_metrics_with_target + candidate_metrics_with_target + [pearson_difference_wrt_target] + [difference_in_numbers_of_rows]
        
    def generate_features(self, query_individual_metrics=[], candidate_individual_metrics=[]):
        if not query_individual_metrics:
            feature_factory_query = FeatureFactory(self.get_joined_query_data())
            query_individual_metrics = feature_factory_query.get_individual_metrics(func=max_in_modulus)
        if not candidate_individual_metrics:
            feature_factory_candidate = FeatureFactory(self.get_joined_candidate_data())
            candidate_individual_metrics = feature_factory_candidate.get_individual_metrics(func=max_in_modulus)

        pairwise_metrics = self.compute_pairwise_metrics()
        return np.array(query_individual_metrics + candidate_individual_metrics + pairwise_metrics)
