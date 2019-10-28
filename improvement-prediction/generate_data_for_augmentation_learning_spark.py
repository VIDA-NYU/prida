#!/usr/bin/env python3
from augmentation_instance import *
import json
from pyspark import SparkConf, SparkContext
import time
from util.file_manager import *
from util.instance_parser import *


def generate_learning_instance(learning_data_record, params):
    """Generates features for each line of the training data and
    the respective learning targets (y).
    """

    # parameters
    cluster_execution = params['cluster']
    hdfs_address = params['hdfs_address']
    hdfs_user = params['hdfs_user']

    # parsing instance and generating features and learning targets
    augmentation_instance = parse_augmentation_instance(
        learning_data_record,
        use_hdfs=cluster_execution,
        hdfs_address=hdfs_address,
        hdfs_user=hdfs_user
    )
    features = augmentation_instance.generate_features()
    target_r2_gain = augmentation_instance.compute_gain_in_r2_score()

    # query and candidate information
    query_dataset = augmentation_instance.get_query_filename()
    candidate_dataset = augmentation_instance.get_candidate_filename()
    target = augmentation_instance.get_target_name()

    return [[query_dataset, target, candidate_dataset] + list(features) + [target_r2_gain]]


if __name__ == '__main__':

    start_time = time.time()

    # Spark context
    conf = SparkConf().setAppName("Feature Generation")
    sc = SparkContext(conf=conf)

    # parameters
    params = json.load(open('.params_feature_generation.json'))
    learning_data_filename = params['learning_data_filename']
    augmentation_learning_data_filename = params['augmentation_learning_data_filename']
    cluster_execution = params['cluster']
    hdfs_address = params['hdfs_address']
    hdfs_user = params['hdfs_user']

    # opening training data file
    if not cluster_execution:
        learning_data_filename = 'file://' + learning_data_filename
    learning_data = sc.textFile(learning_data_filename)

    # generating learning instances
    learning_instances = learning_data.flatMap(
        lambda x: generate_learning_instance(x, params)
    ).map(
        lambda x: ','.join([str(item) for item in x])
    )

    save_file(
        augmentation_learning_data_filename,
        '\n'.join(learning_instances.collect()),
        cluster_execution,
        hdfs_address,
        hdfs_user
    )

    print('Duration: %.4f seconds' % (time.time() - start_time))
