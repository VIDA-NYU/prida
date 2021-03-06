#!/usr/bin/env python3
from augmentation_instance import *
from constants import *
import os

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

def parse_augmentation_instance(file_record):
    """Parses file_record, a JSON instance of the training data in the format
    {'query_data': query_data,
     'query_dataset_id': query_dataset,
     'query_key': query_key
     'target': target,
     'candidate_data': candidate_data,
     'candidate_dataset_id': candidate_dataset,
     'candidate_key': candidate_key,
     'imputation_strategy': imputation_strategy,
     'joined_dataset': joined_dataset,
     'mean_absolute_error': [mae_before, mae_after],
     'mean_squared_error': [mse_before, mse_after],
     'median_absolute_error': [med_ae_before, med_ae_after],
     'r2_score': [r2_score_before, r2_score_after]}

    The parsed file_record is used to generate an AugmentationInstance, which stores and manages the actual datasets
    """
    #TODO standardize the use of words dataset and filename
    
    fields = {'query_data': file_record['query_data'],
              'query_dataset_id': file_record['query_dataset'],
              'query_key': file_record.get('query_key', None), 
              'target_name': file_record['target'],
              'candidate_data': file_record['candidate_data'],
              'candidate_dataset_id': file_record['candidate_dataset'],
              'candidate_key': file_record.get('candidate_key', None),
              'joined_data': None,
              'joined_dataset_id': None,
              'imputation_strategy': file_record['imputation_strategy'],
              'mark': file_record.get('mark', None)}
    if 'joined_dataset' in file_record:
        fields['joined_dataset_id'] = file_record['joined_dataset']
    if 'joined_data' in file_record:
        fields['joined_data'] = file_record['joined_data']
    if 'mean_absolute_error' in file_record:
        fields['mae_before'] = file_record['mean_absolute_error'][0]
        fields['mae_after'] = file_record['mean_absolute_error'][1]
        fields['mse_before'] = file_record['mean_squared_error'][0]
        fields['mse_after'] = file_record['mean_squared_error'][1]
        fields['med_ae_before'] = file_record['median_absolute_error'][0]
        fields['med_ae_after'] = file_record['median_absolute_error'][1]
        fields['r2_score_before'] = file_record['r2_score'][0]
        fields['r2_score_after'] = file_record['r2_score'][1]
    return AugmentationInstance(fields)
