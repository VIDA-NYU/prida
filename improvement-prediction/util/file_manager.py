#!/usr/bin/env python3
from hdfs import InsecureClient
import os
from constants import *

def read_file(file_path, hdfs_client, use_hdfs=False, hdfs_address=None, hdfs_user=None):
    """Opens a file for read and returns its corresponding content.
    """

    output = None
    if use_hdfs:
        #hdfs_client = InsecureClient(hdfs_address, user=hdfs_user)
        if hdfs_client.status(file_path, strict=False):
            with hdfs_client.read(file_path) as reader:
                output = reader.read().decode()
    else:
        if os.path.exists(file_path):
            with open(file_path) as reader:
                output = reader.read()
    return output


def save_file(file_path, content, hdfs_client, use_hdfs=False, hdfs_address=None, hdfs_user=None):
    """Opens a file for write and returns its corresponding file object.
    """

    if use_hdfs:
        #hdfs_client = InsecureClient(hdfs_address, user=hdfs_user)
        if hdfs_client.status(file_path, strict=False):
            print('[WARNING] File already exists: %s' % file_path)
        with hdfs_client.write(file_path) as writer:
            writer.write(content.encode())
    else:
        if os.path.exists(file_path):
            print('[WARNING] File already exists: %s' % file_path)
        with open(file_path, 'w') as writer:
            writer.write(content)
    print('[INFO] File %s saved!' % file_path)

def dump_learning_instances(data_filename, features, targets):
    """For each training instance, this method dumps their corresponding features and 
    targets (relative gains in performance after augmentation) in a file
    """
    with open(data_filename, 'w') as f:
        for features, target in zip(features, targets):
            output_string = ','.join([str(i) for i in features]) + ',' + str(target) + '\n'
            f.write(output_string)

def read_augmentation_learning_filename(augmentation_learning_filename):
    """Reads augmentation_learning_filename to extract metadata, features and targets 
    (relative performance gains after augmentation)
    """
    learning_metadata = []
    learning_features = []
    learning_targets = []
    with open(augmentation_learning_filename, 'r') as f:
        for line in f:
            fields = line.strip().split(SEPARATOR)
            metadata = {'query_filename': fields[QUERY_FILENAME_ID], 'target_name': fields[TARGET_NAME_ID], 'candidate_filename': fields[CANDIDATE_FILENAME_ID]}
            learning_metadata.append(metadata)
            features = [float(i) for i in fields[CANDIDATE_FILENAME_ID+1:DECREASE_IN_MEAN_ABSOLUTE_ERROR_ID]]
            learning_features.append(features)
            targets = {'decrease_in_mae': float(fields[DECREASE_IN_MEAN_ABSOLUTE_ERROR_ID]),
                       'decrease_in_mse': float(fields[DECREASE_IN_MEAN_SQUARED_ERROR_ID]),
                       'decrease_in_med_ae': float(fields[DECREASE_IN_MEDIAN_ABSOLUTE_ERROR_ID]),
                       'gain_in_r2_score': float(fields[GAIN_IN_R2_SCORE_ID])}
            learning_targets.append(targets)
    return learning_metadata, learning_features, learning_targets
