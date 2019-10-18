from dataset import *
import numpy as np
class AugmentationInstance:
    def __init__(self, instance_values):
        self.query_filename = instance_values['query_filename']
        self.query_dataset = Dataset(self.query_filename)
        self.candidate_filename = instance_values['candidate_filename']
        self.candidate_dataset = Dataset(self.candidate_filename)
        self.target_name = instance_values['target_name']
        self.initial_r2_score = instance_values['initial_r2_score']
        self.final_r2_score = instance_values['final_r2_score']
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

    def get_joined_data(self):
        return self.joined_dataset.get_data()

    def get_target_column_name(self):
        return self.target_name

    def compute_r2_gain(self):
        return (self.final_r2_score - self.initial_r2_score)/np.fabs(self.initial_r2_score)
