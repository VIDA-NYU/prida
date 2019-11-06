#!/usr/bin/env python3
from augmentation_instance import *
import json
from pyspark import SparkConf, SparkContext
import time
from util.file_manager import *
from util.instance_parser import *


def generate_learning_instance(prefix, learning_data_record, params):
    """Generates features for each JSON record of the training data
    """

    # parameters
    cluster_execution = params['cluster']
    hdfs_address = params['hdfs_address']
    hdfs_user = params['hdfs_user']

    # creating a global hdfs client for reading and writing purposed
    hdfs_client = InsecureClient(hdfs_address, user=hdfs_user)
    # parsing instance and generating features and learning targets
    augmentation_instance = parse_augmentation_instance(
        prefix, 
        learning_data_record,
        hdfs_client,
        use_hdfs=cluster_execution,
        hdfs_address=hdfs_address,
        hdfs_user=hdfs_user
    )
    features = augmentation_instance.generate_features()
    target_mae_decrease = augmentation_instance.compute_decrease_in_mean_absolute_error()
    target_mse_decrease = augmentation_instance.compute_decrease_in_mean_squared_error()
    target_med_ae_decrease = augmentation_instance.compute_decrease_in_median_absolute_error()
    target_r2_gain = augmentation_instance.compute_gain_in_r2_score()

    # query and candidate information
    query_dataset = augmentation_instance.get_query_filename()
    candidate_dataset = augmentation_instance.get_candidate_filename()
    target = augmentation_instance.get_target_name()

    return [[query_dataset, target, candidate_dataset] + list(features) + [target_mae_decrease, target_mse_decrease, target_med_ae_decrease, target_r2_gain]]


if __name__ == '__main__':

    start_time = time.time()

    # Spark context
    conf = SparkConf().setAppName('Feature Generation')
    sc = SparkContext(conf=conf)

    # parameters
    params = json.load(open('.params_feature_generation.json'))
    learning_data_filename = params['learning_data_filename']
    augmentation_learning_data_filename = params['augmentation_learning_data_filename']
    cluster_execution = params['cluster']
    hdfs_address = params['hdfs_address']
    hdfs_user = params['hdfs_user']

    # opening training data file
    learning_data_filename_for_spark = learning_data_filename
    if not cluster_execution:
        learning_data_filename_for_spark = 'file://' + learning_data_filename
    learning_data = sc.textFile(learning_data_filename_for_spark).repartition(NUMBER_OF_SPARK_REPARTITIONS)

    # generating learning instances
    learning_instances = learning_data.flatMap(
        lambda x: generate_learning_instance(get_prefix_of_training_files(learning_data_filename), json.loads(x), params)
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
