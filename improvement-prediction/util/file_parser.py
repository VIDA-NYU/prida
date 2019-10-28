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
    """Parses file_record, an instance of the training data composed of a query filename, a target (a column in the query 
    dataset whose values should be predicted), a candidate filename, and different metrics regarding the prediction of the 
    target with the query dataset alone and with the final dataset (query augmented with candidate). The parsed file_record 
    is used to generate an AugmentationInstance, which stores and manages the actual datasets
    """
    query_filename, target, candidate_filename, r2_score_before, r2_score_after = file_record.strip().split(SEPARATOR)
    fields = {'query_filename': query_filename,
              'target_name': target,
              'candidate_filename': candidate_filename,
              'r2_score_before': float(r2_score_before),
              'r2_score_after': float(r2_score_after)}
    return AugmentationInstance(fields, use_hdfs, hdfs_address, hdfs_user)
