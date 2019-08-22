from dataset import *

class AugmentationInstance:
    def __init__(self, instance_values):
        self.query_filename = instance_values['query_filename']
        self.query_dataset = Dataset(self.query_filename)
        self.candidate_filename = instance_values['candidate_filename']
        self.candidate_dataset = Dataset(self.candidate_filename)
        self.target_name = instance_values['target_name']
        self.initial_r2_score = instance_values['initial_r2_score']
        self.final_r2_score = instance_values['final_r2_score']
        
    def get_query_dataset(self):
        return self.query_dataset

    def get_candidate_dataset(self):
        return self.candidate_dataset
