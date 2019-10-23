import numpy as np
from dataset import *
from feature_factory import *

class AugmentationInstance:
    def __init__(self, instance_values):
        self.query_filename = instance_values['query_filename']
        self.query_dataset = Dataset(self.query_filename)
        self.candidate_filename = instance_values['candidate_filename']
        self.candidate_dataset = Dataset(self.candidate_filename)
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
        return Dataset(result_data, result_data.columns)

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

    def compute_r2_gain(self):
        return (self.r2_score_after - self.r2_score_before)/np.fabs(self.r2_score_before)

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

        return fd_metrics + metrics_with_target + query_metrics_with_target + candidate_metrics_with_target + [pearson_difference_wrt_target]
        

    def generate_features(self, query_individual_metrics=[], candidate_individual_metrics=[]):
        if not query_individual_metrics:
            feature_factory_query = FeatureFactory(self.get_joined_query_data())
            query_individual_metrics = feature_factory_query.get_individual_metrics(func=max_in_modulus)
        if not candidate_individual_metrics:
            feature_factory_candidate = FeatureFactory(self.get_joined_candidate_data())
            candidate_individual_metrics = feature_factory_candidate.get_individual_metrics(func=max_in_modulus)

        pairwise_metrics = self.compute_pairwise_metrics()
        return query_individual_metrics + candidate_individual_metrics + pairwise_metrics

#         full_dataset_pairwise_metrics, pairwise_metrics_with_target, candidate_pairwise_metrics_with_target, query_pairwise_metrics_with_target, pearson_difference_wrt_target


#                 # TODO encapsulate below
#                 feature_factory_full_dataset = FeatureFactory(instance.get_joined_data())
#                 full_dataset_pairwise_metrics = feature_factory_full_dataset.get_pairwise_metrics(func=max_in_modulus)
#                 pairwise_metrics_with_target = feature_factory_full_dataset.get_pairwise_metrics_with_target(instance.get_target_column_name(),
#                                                                                                              func=max_in_modulus)
#                 feature_factory_candidate_with_target = FeatureFactory(instance.get_joined_candidate_data_and_target())
#                 candidate_pairwise_metrics_with_target = feature_factory_candidate_with_target.get_pairwise_metrics_with_target(instance.get_target_column_name(),
#                                                                                                                                 func=max_in_modulus)
#                 feature_factory_query = FeatureFactory(instance.get_joined_query_data())
#                 query_pairwise_metrics_with_target = feature_factory_query.get_pairwise_metrics_with_target(instance.get_target_column_name(),
#                                                                                                             func=max_in_modulus)

#                 pearson_difference_wrt_target = feature_factory_candidate_with_target.compute_difference_in_pearsons_wrt_target(
#                     feature_factory_query.get_max_pearson_wrt_target(instance.get_target_column_name()), instance.get_target_column_name())

#                 learning_features = self.query_individual_metrics[query_filename]
#                 print('len(learning_features) =', len(learning_features))
#                 learning_features += self.candidate_individual_metrics[candidate]
#                 print('len(learning_features) =', len(learning_features))
#                 learning_features += full_dataset_pairwise_metrics 
#                 print('len(learning_features) =', len(learning_features))
#                 learning_features += pairwise_metrics_with_target
#                 print('len(learning_features) =', len(learning_features))
#                 learning_features += query_pairwise_metrics_with_target
#                 print('len(learning_features) =', len(learning_features))
#                 learning_features += candidate_pairwise_metrics_with_target
#                 print('len(learning_features) =', len(learning_features))
#                 learning_features += [pearson_difference_wrt_target]
#                 print('len(learning_features) =', len(learning_features))




#         feature_factory_full_dataset = FeatureFactory(instance.get_joined_data())
#         full_dataset_pairwise_metrics = feature_factory_full_dataset.get_pairwise_metrics(func=max_in_modulus)
#         pairwise_metrics_with_target = feature_factory_full_dataset.get_pairwise_metrics_with_target(instance.get_target_column_name(),
#                                                                                                      func=max_in_modulus)
#         query_pairwise_metrics_with_target = feature_factory_query.get_pairwise_metrics_with_target(instance.get_target_column_name(),
#                                                                                                     func=max_in_modulus)
#         feature_factory_candidate_with_target = FeatureFactory(instance.get_joined_candidate_data_and_target())
#         candidate_pairwise_metrics_with_target = feature_factory_candidate_with_target.get_pairwise_metrics_with_target(instance.get_target_column_name(),
#                                                                                                                         func=max_in_modulus)
#         max_in_modulus_pearson_difference = feature_factory_candidate_with_target.compute_difference_in_pearsons_wrt_target(feature_factory_query.get_max_pearson_wrt_target(instance.get_target_column_name()), instance.get_target_column_name())
        
#         r2_gain = instance.compute_r2_gain()

#         learning_features = query_individual_metrics + \
#                             candidate_individual_metrics + \
#                             full_dataset_pairwise_metrics + \
#                             pairwise_metrics_with_target + \
#                             query_pairwise_metrics_with_target + \
#                             candidate_pairwise_metrics_with_target + \
#                             [max_in_modulus_pearson_difference]
