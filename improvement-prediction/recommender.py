import pandas as pd
from constants import *
from augmentation_instance import *
from feature_factory import *
from learning_task import *

class Recommender:
    
    def store_instances(self, filename):

        with open(filename, 'r') as f:
            rows_list = []
            self.candidate_filenames = []
            self.query_individual_metrics = {}
            self.candidate_individual_metrics = {}
            for line in f:
                query_filename, target_name, candidate_filename, r2_score_before, r2_score_after = line.strip().split(SEPARATOR)
                self.candidate_filenames.append(candidate_filename)

                fields = {'query_filename': query_filename,
                            'target_name': target_name,
                            'candidate_filename': candidate_filename,
                            'r2_score_before': float(r2_score_before),
                            'r2_score_after': float(r2_score_after)}
                rows_list.append(fields)

                instance = AugmentationInstance(fields)
                self.query_individual_metrics[query_filename] = \
                    FeatureFactory(instance.get_joined_query_data()).get_individual_metrics(func=max_in_modulus)
                self.candidate_individual_metrics[candidate_filename] = \
                    FeatureFactory(instance.get_joined_candidate_data()).get_individual_metrics(func=max_in_modulus)

            self.data = pd.DataFrame(rows_list) 
            self.data.set_index(['query_filename', 'target_name', 'candidate_filename'])

    def generate_models_and_test_data(self, augmentation_learning_data_filename, n_splits):
        self.learning_task = LearningTask()
        self.learning_task.read_data(augmentation_learning_data_filename)
        return self.learning_task.execute_random_forest(n_splits)

    def recommend_candidates(self, model, data):
        for index in data['index_of_test_instances']:
            query_filename = self.data.iloc[index]['query_filename']
            target_name = self.data.iloc[index]['target_name']
            for candidate in self.candidate_filenames:
                instance = AugmentationInstance({'query_filename': query_filename,
                                                 'target_name': target_name,
                                                 'candidate_filename': candidate})
                # TODO avoid computing feature_factory_query more than once
                test_features = instance.generate_features(self.query_individual_metrics[query_filename], 
                                           self.candidate_individual_metrics[candidate])


                ##
                print('predicted gain for candidate', candidate, 'is', model.predict(test_features).reshape(1, -1)))
                break
            break
