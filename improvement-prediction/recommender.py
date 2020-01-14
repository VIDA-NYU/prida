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
        r_precision = []
        avg_precs = []
        ndcgs = []
        
        kendall_tau_containment_baseline = []
        precision_at_1_containment_baseline = []
        precision_at_5_containment_baseline = []
        r_precision_containment_baseline = []
        avg_precs_containment_baseline = []
        ndcgs_containment_baseline = []
        
        kendall_tau_difference_in_pearson_baseline = []
        precision_at_1_difference_in_pearson_baseline = []
        precision_at_5_difference_in_pearson_baseline = []
        r_precision_difference_in_pearson_baseline = []
        avg_precs_difference_in_pearson_baseline = []
        ndcgs_difference_in_pearson_baseline = []
        
        number_of_candidates_per_query_target = []
        for query_target in gains_dict.keys():
            query_target_gains = gains_dict[query_target]
            real_gains = [(i, query_target_gains[i][REAL_GAIN_ID]) for i in query_target_gains]
            number_of_candidates_per_query_target.append(len(real_gains))
            predicted_gains = [(i, query_target_gains[i][PREDICTED_GAIN_ID]) for i in query_target_gains]
            containment_gains = [(i, query_target_gains[i][CONTAINMENT_BASELINE_GAIN_ID]) for i in query_target_gains]
            diff_pearson_gains = [(i, query_target_gains[i][DIFF_PEARSON_BASELINE_GAIN_ID]) for i in query_target_gains]

            precision_at_1.append(compute_r_precision(real_gains, predicted_gains, k=1))
            precision_at_5.append(compute_r_precision(real_gains, predicted_gains))
            avg_precs.append(compute_average_precision(real_gains, predicted_gains))
            ndcgs.append(compute_ndcg_at_k(real_gains, predicted_gains))
            
            precision_at_1_containment_baseline.append(compute_r_precision(real_gains, containment_gains, k=1))
            precision_at_5_containment_baseline.append(compute_r_precision(real_gains, containment_gains))
            avg_precs_containment_baseline.append(compute_average_precision(real_gains, containment_gains))
            ndcgs_containment_baseline.append(compute_ndcg_at_k(real_gains, containment_gains))
            
            precision_at_1_difference_in_pearson_baseline.append(compute_r_precision(real_gains, diff_pearson_gains, k=1))
            precision_at_5_difference_in_pearson_baseline.append(compute_r_precision(real_gains, diff_pearson_gains))
            avg_precs_difference_in_pearson_baseline.append(compute_average_precision(real_gains, diff_pearson_gains))
            ndcgs_difference_in_pearson_baseline.append(compute_ndcg_at_k(real_gains, diff_pearson_gains))
            
            kendalltau = compute_kendall_tau(real_gains, predicted_gains)[0]
            kendalltau_containment = compute_kendall_tau(real_gains, containment_gains)[0]
            kendalltau_pearson = compute_kendall_tau(real_gains, diff_pearson_gains)[0]
            if not np.isnan(kendalltau) and not np.isnan(kendalltau_containment) and not np.isnan(kendalltau_pearson):
                kendall_tau.append(kendalltau)
                kendall_tau_containment_baseline.append(kendalltau_containment)
                kendall_tau_difference_in_pearson_baseline.append(kendalltau_pearson)

            rprecision = compute_r_precision(real_gains, predicted_gains, positive_only=True)
            rprecision_containment = compute_r_precision(real_gains, containment_gains, positive_only=True)
            rprecision_pearson = compute_r_precision(real_gains, diff_pearson_gains, positive_only=True)
            if not np.isnan(rprecision) and not np.isnan(rprecision_containment) and not np.isnan(rprecision_pearson):
                r_precision.append(rprecision)
                r_precision_containment_baseline.append(rprecision_containment)
                r_precision_difference_in_pearson_baseline.append(rprecision_pearson)
        
            #if len(number_of_candidates_per_query_target) == 50:
            #    break
            
        print('average number of candidates per query-target', np.mean(number_of_candidates_per_query_target), 'total number of times we could compute kendalltau', len(kendall_tau))
        print('average kendall tau:', np.mean(kendall_tau), 'average kendall tau - containment baseline:', np.mean(kendall_tau_containment_baseline), 'average kendall tau - difference_in_pearson baseline:', np.mean(kendall_tau_difference_in_pearson_baseline))
        print('average precision at 1:', np.mean(precision_at_1), 'average precision at 1 - containment baseline:', np.mean(precision_at_1_containment_baseline), 'average precision at 1 - difference_in_pearson baseline:', np.mean(precision_at_1_difference_in_pearson_baseline))
        print('average precision at 5:', np.mean(precision_at_5), 'average precision at 5 - containment baseline:', np.mean(precision_at_5_containment_baseline), 'average precision at 5 - difference_in_pearson baseline:', np.mean(precision_at_5_difference_in_pearson_baseline))
        print('average R-precision:', np.mean(r_precision), 'average R-precision - containment baseline:', np.mean(r_precision_containment_baseline), 'average R-precision - difference_in_pearson baseline:', np.mean(r_precision_difference_in_pearson_baseline))
        print('MAP:', np.mean(avg_precs), 'MAP - containment baseline:', np.mean(avg_precs_containment_baseline), 'MAP - difference_in_pearson baseline:', np.mean(avg_precs_difference_in_pearson_baseline))
        print('NDCGs:', np.mean(ndcgs), 'NDCGs - containment baseline:', np.mean(ndcgs_containment_baseline), 'NDCGs - difference_in_pearson baseline:', np.mean(ndcgs_difference_in_pearson_baseline))
