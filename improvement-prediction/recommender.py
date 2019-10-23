import pandas as pd
from constants import *
from augmentation_instance import *
from learning_task import *

class Recommender:
    
    def store_instances(self, filename):
        with open(filename, 'r') as f:
            rows_list = []
            self.augmentation_instances = []
            for line in f:
                query_filename, target_name, candidate_filename, r2_score_before, r2_score_after = line.strip().split(SEPARATOR)
                fields = {'query_filename': query_filename,
                            'target_name': target_name,
                            'candidate_filename': candidate_filename,
                            'r2_score_before': float(r2_score_before),
                            'r2_score_after': float(r2_score_after)}
                rows_list.append(fields)
                self.augmentation_instances.append(AugmentationInstance(fields))
            self.data = pd.DataFrame(rows_list) 
            self.data.set_index(['query_filename', 'target_name', 'candidate_filename'])

    def generate_models_and_test_data(self, augmentation_learning_data_filename, n_splits):
        self.learning_task = LearningTask()
        self.learning_task.read_data(augmentation_learning_data_filename)
        return self.learning_task.execute_random_forest(n_splits)

    def recommend_candidates(self, model, data):
        print(self.data.iloc[0])
        # for index in data['index_of_test_instances']:
            
        #     break
