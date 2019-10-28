#!/usr/bin/env python3
from hdfs import InsecureClient
import os


def read_file(file_path, use_hdfs=False, hdfs_address=None, hdfs_user=None):
    """Opens a file for read and returns its corresponding content.
    """

    output = None
    if use_hdfs:
        hdfs_client = InsecureClient(hdfs_address, user=hdfs_user)
        if hdfs_client.status(file_path, strict=False):
            with hdfs_client.read(file_path) as reader:
                output = reader.read().decode()
    else:
        if os.path.exists(file_path):
            with open(file_path) as reader:
                output = reader.read()
    return output


def save_file(file_path, content, use_hdfs=False, hdfs_address=None, hdfs_user=None):
    """Opens a file for write and returns its corresponding file object.
    """

    if use_hdfs:
        hdfs_client = InsecureClient(hdfs_address, user=hdfs_user)
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
