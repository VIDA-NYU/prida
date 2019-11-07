import pandas as pd
import json
import pickle
from augmentation_instance import *
from feature_factory import *
from learning_task import *
from util.metrics import *
from util.instance_parser import *
from constants import *

class Recommender:
    def __init__(self, learning_data_filename):
        """This class (1) generates features and relative gains to be predicted,
        given a training data filename, (2) creates machine learning models to predict the 
        relative gain in performance for augmentation with different candidates, and 
        (3) recommends such candidates sorted by their predicted relative gains 
        """
        self.learning_data_filename = learning_data_filename
        self.prefix = get_prefix_of_training_files(self.learning_data_filename)
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
                    FeatureFactory(instance.get_joined_query_data()).get_individual_features(func=max_in_modulus)
                self.candidate_individual_features[instance.get_candidate_filename()] = \
                    FeatureFactory(instance.get_joined_candidate_data()).get_individual_features(func=max_in_modulus)

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
        instances for it (.json format), this method loads both models and test instances
        """
        models = []
        test_data = []
        with open(augmentation_models_and_tests_filename, 'r') as f:
            for line in f:
                model_filename, test_filename = line.strip().split(SEPARATOR)
                models.append(pickle.load(open(model_filename, 'rb')))
                test_data.append(json.load(open(test_filename, 'r')))
        return models, test_data

    def get_real_and_predicted_gains(self, query_filename, target_name, model, metric='r2_score'):
        """Given the names of a query dataset and a target (a column in the query dataset), 
        this method predicts the relative gain obtained via data augmentation with a variety of 
        candidate datasets, then returning these predicted gains and the corresponding real ones, 
        which are stored in a table
        """
        subtable = self.learning_table[(self.learning_table['query_filename'] == query_filename) & 
                                       (self.learning_table['target_name'] == target_name)]
        predicted_gains = []
        real_gains = []

        for index, row in subtable.iterrows():
            candidate_filename = row['candidate_filename']
            if metric == 'r2_score':
                real_gains.append((candidate_filename, compute_r2_gain(row['r2_score_before'], row['r2_score_after'])))
            elif metric == 'mean_absolute_error':
                real_gains.append((candidate_filename, compute_mae_decrease(row['mae_before'], row['mae_after'])))
            elif metric == 'mean_squared_error':
                real_gains.append((candidate_filename, compute_mse_decrease(row['mse_before'], row['mse_after'])))
            else:
                real_gains.append((candidate_filename, compute_med_ae_decrease(row['med_ae_before'], row['med_ae_after'])))

            # we test joining query and candidate datasets with different types
            # of imputation, as they are hidden in test time
            test_instance_mean = AugmentationInstance({'query_filename': query_filename,
                                                       'target_name': target_name,
                                                       'candidate_filename': candidate_filename,
                                                       'imputation_strategy': 'mean'})
            test_features_mean = test_instance_mean.generate_features(self.query_individual_features[query_filename], 
                                                                      self.candidate_individual_features[candidate_filename])
            gain_mean = model.predict(test_features_mean.reshape(1, -1))[0]
            
            test_instance_median = AugmentationInstance({'query_filename': query_filename,
                                                         'target_name': target_name,
                                                         'candidate_filename': candidate_filename,
                                                         'imputation_strategy': 'median'})
            test_features_median = test_instance_median.generate_features(self.query_individual_features[query_filename], 
                                                                          self.candidate_individual_features[candidate_filename])
            gain_median = model.predict(test_features_median.reshape(1, -1))[0]
          
            test_instance_most_frequent = AugmentationInstance({'query_filename': query_filename,
                                                                'target_name': target_name,
                                                                'candidate_filename': candidate_filename,
                                                                'imputation_strategy': 'most_frequent'})
            test_features_most_frequent = test_instance_most_frequent.generate_features(self.query_individual_features[query_filename], 
                                                                                        self.candidate_individual_features[candidate_filename])
            gain_most_frequent = model.predict(test_features_most_frequent.reshape(1, -1))[0]

            # we keep the best predicted gain we find with different imputation strategies
            predicted_gains.append((candidate_filename, max([gain_mean, gain_median, gain_most_frequent])))
        return real_gains, predicted_gains
        
    def predict_gains_for_candidate_datasets(self, model, data):
        """This method encapsulates the prediction of relative gains via data augmentation using a given 
        machine learning model and test data
        """
        for index in data['index_of_test_instances']:
            query_filename = self.learning_table.iloc[index]['query_filename']
            target_name = self.learning_table.iloc[index]['target_name']
            real_gains, predicted_gains = self.get_real_and_predicted_gains(query_filename, target_name, model)
            print('REAL GAINS', real_gains)
            print('PREDICTED GAINS', predicted_gains)
            print(compute_ndcg_at_k(real_gains, predicted_gains, use_gains_as_relevance_weights=True))
            print(compute_kendall_tau(real_gains, predicted_gains))
            print(compute_mean_reciprocal_rank_for_single_sample(real_gains, predicted_gains))
            #TODO compute average for all mean_reciprocal_ranks outside this loop. should i limit the mrr to k=5?
            break
