import pandas as pd
import json
import pickle
from augmentation_instance import *
from feature_factory import *
from learning_task import *
from util.metrics import *
from util.instance_parser import *
from util.file_manager import *
from constants import *

class Recommender:
    def __init__(self, learning_data_filename, store_instances=True):
        """This class (1) generates features and relative gains to be predicted,
        given a training data filename, (2) creates machine learning models to predict the 
        relative gain in performance for augmentation with different candidates, and 
        (3) recommends such candidates sorted by their predicted relative gains 
        """
        self.learning_data_filename = learning_data_filename
        self.prefix = get_prefix_of_training_files(self.learning_data_filename)

        if store_instances:
            self._store_instances()
        
    def _store_instances(self):
        """Given a training data filename, this method derives their corresponding 
        features and relative gains (targets), stores individual features for query and 
        candidate datasets for efficiency, so they don't need to be recomputed, and 
        stores the rows of the filename in a table
        """
        with open(self.learning_data_filename, 'r') as f:
            rows_list = []
            self.query_individual_features = {}
            self.candidate_individual_features = {}
            for line in f:
                instance = parse_augmentation_instance(self.prefix, json.loads(line))
                rows_list.append(instance.get_formatted_fields())               
                self.query_individual_features[instance.get_query_filename()] = \
                    FeatureFactory(instance.get_query_dataset().get_data()).get_individual_features(func=max_in_modulus)
                self.candidate_individual_features[instance.get_candidate_filename()] = \
                    FeatureFactory(instance.get_candidate_dataset().get_data()).get_individual_features(func=max_in_modulus)

            self.learning_table = pd.DataFrame(rows_list) 
            self.learning_table.set_index(['query_filename', 'target_name', 'candidate_filename'])

    def generate_models_and_test_data(self, augmentation_learning_data_filename, n_splits):
        """Given a filename with features and relative gains for every training instance, 
        this method generates machine learning models and pointers to test data over which the 
        models can be evaluated
        """
        self.learning_task = LearningTask()
        self.learning_task.read_features_and_targets(augmentation_learning_data_filename)
        return self.learning_task.execute_random_forest(n_splits)

    def read_models_and_test_data(self, augmentation_models_and_tests_filename):
        """Given a filename where each line lists a file with a model (.sav format) and test 
        instances for it (.txt format), this method loads the models and keeps track of the 
        test instances' filenames
        """
        models = []
        test_data_filenames = []
        with open(augmentation_models_and_tests_filename, 'r') as f:
            for line in f:
                model_filename, test_filename = line.strip().split(SEPARATOR)
                models.append(pickle.load(open(model_filename, 'rb')))
                test_data_filenames.append(test_filename)
        return models, test_data_filenames

    def get_real_and_predicted_gains(self, model, data_filename, metric='r2_score'):
        """Given a model and a filename with test instances,  this method predicts the relative gain obtained via data augmentation 
        with a variety of  candidate datasets, then returning these predicted gains and the corresponding real ones, along with a few 
        baseline predicted gains, in a dictionary
        """
        gains_dict = {}
        with open(data_filename) as test_file:
            i = 0
            for line in test_file:
                metadata, features, targets = parse_learning_instance(line)
                dict_key = metadata['query_filename'] + '-' + metadata['target_name']
                candidate_filename = metadata['candidate_filename']
                if metric == 'r2_score':
                    real_gain = targets['gain_in_r2_score']
                elif metric == 'mean_absolute_error':
                    real_gain = targets['decrease_in_mae']
                elif metric == 'mean_squared_error':
                    real_gain = targets['decrease_in_mse']
                else:
                    real_gain = targets['decrease_in_med_ae']
                predicted_gain = model.predict(np.array(features).reshape(1, -1))[0]
                containment_baseline_gain = features[CONTAINMENT_FEATURE_ID]
                difference_in_pearson_baseline_gain = features[DIFFERENCE_IN_PEARSON_FEATURE_ID]

                if dict_key in gains_dict:
                    gains_dict[dict_key][candidate_filename] = [real_gain, predicted_gain, containment_baseline_gain, difference_in_pearson_baseline_gain]
                else:
                    gains_dict[dict_key] = {candidate_filename: [real_gain, predicted_gain, containment_baseline_gain, difference_in_pearson_baseline_gain]}
                i += 1
        return gains_dict

    def generate_and_evaluate_predicted_gains(self, model, data):
        """This method encapsulates the prediction of relative gains via data augmentation using a given 
        machine learning model and test data
        """
        gains_dict = self.get_real_and_predicted_gains(model, data)
        print('number of query filename + target combinations', len(gains_dict.keys()))
        
        kendall_tau = []
        precision_at_1 = []
        precision_at_5 = []
        kendall_tau_containment_baseline = []
        precision_at_1_containment_baseline = []
        precision_at_5_containment_baseline = []
        kendall_tau_difference_in_pearson_baseline = []
        precision_at_1_difference_in_pearson_baseline = []
        precision_at_5_difference_in_pearson_baseline = []

        number_of_candidates_per_query_target = []
        for query_target in gains_dict.keys():
            query_target_gains = gains_dict[query_target]
            real_gains = [(i, query_target_gains[i][REAL_GAIN_ID]) for i in query_target_gains]
            number_of_candidates_per_query_target.append(len(real_gains))
            predicted_gains = [(i, query_target_gains[i][PREDICTED_GAIN_ID]) for i in query_target_gains]
            containment_gains = [(i, query_target_gains[i][CONTAINMENT_BASELINE_GAIN_ID]) for i in query_target_gains]
            diff_pearson_gains = [(i, query_target_gains[i][DIFF_PEARSON_BASELINE_GAIN_ID]) for i in query_target_gains]
            kendall_tau.append(compute_kendall_tau(real_gains, predicted_gains)[0])
            precision_at_1.append(compute_precision_at_k(real_gains, predicted_gains, k=1))
            precision_at_5.append(compute_precision_at_k(real_gains, predicted_gains))
            kendall_tau_containment_baseline.append(compute_kendall_tau(real_gains, containment_gains)[0])
            precision_at_1_containment_baseline.append(compute_precision_at_k(real_gains, containment_gains, k=1))
            precision_at_5_containment_baseline.append(compute_precision_at_k(real_gains, containment_gains))
            kendall_tau_difference_in_pearson_baseline.append(compute_kendall_tau(real_gains, diff_pearson_gains)[0])
            precision_at_1_difference_in_pearson_baseline.append(compute_precision_at_k(real_gains, diff_pearson_gains, k=1))
            precision_at_5_difference_in_pearson_baseline.append(compute_precision_at_k(real_gains, diff_pearson_gains))
        print('lengths', len(precision_at_1), len(precision_at_1_containment_baseline), len(precision_at_1_difference_in_pearson_baseline))
        print('average number of candidates per query-target', np.mean(number_of_candidates_per_query_target))
        print('average kendall tau:', np.mean(kendall_tau), 'average kendall tau - containment baseline:', np.mean(kendall_tau_containment_baseline), 'average kendall tau - difference_in_pearson baseline:', np.mean(kendall_tau_difference_in_pearson_baseline))
        print('average precision at 1:', np.mean(precision_at_1), 'average precision at 1 - containment baseline:', np.mean(precision_at_1_containment_baseline), 'average precision at 1 - difference_in_pearson baseline:', np.mean(precision_at_1_difference_in_pearson_baseline))
        print('average precision at 5:', np.mean(precision_at_5), 'average precision at 5 - containment baseline:', np.mean(precision_at_5_containment_baseline), 'average precision at 5 - difference_in_pearson baseline:', np.mean(precision_at_5_difference_in_pearson_baseline))


    # def read_models_and_test_data(self, augmentation_models_and_tests_filename):
    #     """Given a filename where each line lists a file with a model (.sav format) and test 
    #     instances for it (.json format), this method loads both models and test instances
    #     """
    #     models = []
    #     test_data = []
    #     with open(augmentation_models_and_tests_filename, 'r') as f:
    #         for line in f:
    #             model_filename, test_filename = line.strip().split(SEPARATOR)
    #             models.append(pickle.load(open(model_filename, 'rb')))
    #             test_data.append(json.load(open(test_filename, 'r')))
    #     return models, test_data

    # def get_real_and_predicted_gains(self, query_filename, target_name, model, metric='r2_score'):
    #     """Given the names of a query dataset and a target (a column in the query dataset), 
    #     this method predicts the relative gain obtained via data augmentation with a variety of 
    #     candidate datasets, then returning these predicted gains and the corresponding real ones, 
    #     which are stored in a table
    #     """
    #     subtable = self.learning_table[(self.learning_table['query_filename'] == query_filename) & 
    #                                    (self.learning_table['target_name'] == target_name)]
    #     try:
    #         subtable = subtable.sample(n=20, random_state=42)
    #     except:
    #         print('For given query and target_name, there are fewer than 20 candidate datasets')
        
    #     predicted_gains = []
    #     real_gains = []

    #     # - containment_baseline_gains uses the maximum key intersection between query and candidate
    #     #   datasets
    #     # - difference_in_pearson_baseline_gains uses the difference between the max_in_modulus pearson
    #     #   correlation considering query columns and target, and candidate columns and target
    #     #TODO encapsulate these baselines
    #     containment_baseline_gains = []
    #     difference_in_pearson_baseline_gains = []
    #     for index, row in subtable.iterrows():
    #         candidate_filename = row['candidate_filename']
    #         if metric == 'r2_score':
    #             real_gains.append((candidate_filename, compute_r2_gain(row['r2_score_before'], row['r2_score_after'])))
    #         elif metric == 'mean_absolute_error':
    #             real_gains.append((candidate_filename, compute_mae_decrease(row['mae_before'], row['mae_after'])))
    #         elif metric == 'mean_squared_error':
    #             real_gains.append((candidate_filename, compute_mse_decrease(row['mse_before'], row['mse_after'])))
    #         else:
    #             real_gains.append((candidate_filename, compute_med_ae_decrease(row['med_ae_before'], row['med_ae_after'])))

    #         # we test joining query and candidate datasets with different types
    #         # of imputation, as they are hidden in test time
    #         test_instance_mean = AugmentationInstance({'query_filename': query_filename,
    #                                                    'target_name': target_name,
    #                                                    'candidate_filename': candidate_filename,
    #                                                    'imputation_strategy': 'mean'})
    #         test_features_mean = test_instance_mean.generate_features(self.query_individual_features[query_filename], 
    #                                                                   self.candidate_individual_features[candidate_filename])
    #         gain_mean = model.predict(test_features_mean.reshape(1, -1))[0]
            
    #         predicted_gains.append((candidate_filename, gain_mean))            
    #         # there are features already computed in test_features_mean that actually correspond to the two baselines we are
    #         # experimenting with here.
    #         containment_baseline_gains.append((candidate_filename, test_features_mean[CONTAINMENT_FEATURE_ID]))
    #         difference_in_pearson_baseline_gains.append((candidate_filename, test_features_mean[DIFFERENCE_IN_PEARSON_FEATURE_ID]))
    #     return real_gains, predicted_gains, containment_baseline_gains, difference_in_pearson_baseline_gains

    # def predict_gains_for_candidate_datasets(self, model, data):
    #     """This method encapsulates the prediction of relative gains via data augmentation using a given 
    #     machine learning model and test data
    #     """
    #     i = 0
    #     kendall_tau = []
    #     precision_at_1 = []
    #     precision_at_5 = []
    #     #precision_at_50 = []
    #     kendall_tau_containment_baseline = []
    #     precision_at_1_containment_baseline = []
    #     precision_at_5_containment_baseline = []
    #     kendall_tau_difference_in_pearson_baseline = []
    #     precision_at_1_difference_in_pearson_baseline = []
    #     precision_at_5_difference_in_pearson_baseline = []
    #     for index in data['index_of_test_instances']:
    #         query_filename = self.learning_table.iloc[index]['query_filename']
    #         target_name = self.learning_table.iloc[index]['target_name']
    #         real, predicted, containment_baseline, difference_in_pearson_baseline = self.get_real_and_predicted_gains(query_filename, target_name, model)
    #         kendall_tau.append(compute_kendall_tau(real, predicted)[0])
    #         precision_at_1.append(compute_precision_at_k(real, predicted, k=1))
    #         prec = compute_precision_at_k(real, predicted)
    #         precision_at_5.append(prec)
    #         if prec < 0.2:
    #             print('precision at 5 for index', index, 'is', prec)
    #         kendall_tau_containment_baseline.append(compute_kendall_tau(real, containment_baseline)[0])
    #         precision_at_1_containment_baseline.append(compute_precision_at_k(real, containment_baseline, k=1))
    #         precision_at_5_containment_baseline.append(compute_precision_at_k(real, containment_baseline))
    #         kendall_tau_difference_in_pearson_baseline.append(compute_kendall_tau(real, difference_in_pearson_baseline)[0])
    #         precision_at_1_difference_in_pearson_baseline.append(compute_precision_at_k(real, difference_in_pearson_baseline, k=1))
    #         precision_at_5_difference_in_pearson_baseline.append(compute_precision_at_k(real, difference_in_pearson_baseline))
    #         i += 1
    #         if i == 5:
    #             break
    #     print('lengths', len(precision_at_1), len(precision_at_1_containment_baseline), len(precision_at_1_difference_in_pearson_baseline))
    #     print('average kendall tau:', np.mean(kendall_tau), 'average kendall tau - containment baseline:', np.mean(kendall_tau_containment_baseline), 'average kendall tau - difference_in_pearson baseline:', np.mean(kendall_tau_difference_in_pearson_baseline))
    #     print('average precision at 1:', np.mean(precision_at_1), 'average precision at 1 - containment baseline:', np.mean(precision_at_1_containment_baseline), 'average precision at 1 - difference_in_pearson baseline:', np.mean(precision_at_1_difference_in_pearson_baseline))
    #     print('average precision at 5:', np.mean(precision_at_5), 'average precision at 5 - containment baseline:', np.mean(precision_at_5_containment_baseline), 'average precision at 5 - difference_in_pearson baseline:', np.mean(precision_at_5_difference_in_pearson_baseline))
            
