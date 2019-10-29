#!/usr/bin/env python3
from augmentation_instance import *
from constants import *

def parse_augmentation_instances(training_data_filename):
    """If spark is not being used, this method reads every line of 
    the training data (training_data_filename) and invokes their 
    parsing
    """
    with open(filename, 'r') as f:
        augmentation_instances = []
        for line in f:
            augmentation_instances.append(parse_augmentation_instance(line))
        return augmentation_instances


def parse_augmentation_instance(file_record, use_hdfs=False, hdfs_address=None, hdfs_user=None):
    """Parses file_record, a JSON instance of the training data in the format
    {'query_dataset': query_dataset,
     'target': target,
     'candidate_dataset': candidate_dataset,
     'imputation_strategy': imputation_strategy,
     'mean_absolute_error': [mae_before, mae_after],
     'mean_squared_error': [mse_before, mse_after],
     'median_absolute_error': [med_ae_before, med_ae_after],
     'r2_score': [r2_score_before, r2_score_after]}

    The parsed file_record is used to generate an AugmentationInstance, which stores and manages the actual datasets
    """
    #TODO standardize the use of words dataset and filename
    
    fields = {'query_filename': file_record['query_dataset'],
              'target_name': file_record['target'],
              'candidate_filename': file_record['candidate_dataset'],
              'imputation_strategy': file_record['imputation_strategy'], 
              'mae_before': file_record['mean_absolute_error'][0],
              'mae_after': file_record['mean_absolute_error'][1],
              'mse_before': file_record['mean_squared_error'][0],
              'mse_after': file_record['mean_squared_error'][1],
              'med_ae_before': file_record['median_absolute_error'][0],
              'med_ae_after': file_record['median_absolute_error'][1],
              'r2_score_before': file_record['r2_score'][0],
              'r2_score_after': file_record['r2_score'][1]}
    return AugmentationInstance(fields, use_hdfs, hdfs_address, hdfs_user)
