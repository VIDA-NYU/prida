#!/usr/bin/env python3
from augmentation_instance import *
import json
from pyspark import SparkConf, SparkContext
import time
from util.file_manager import *
from util.instance_parser import *


def generate_learning_instance(learning_data_record):
    """Generates features for each JSON record of the training data
    """
    
    # parsing instance and generating features and learning targets
    augmentation_instance = parse_augmentation_instance(learning_data_record)
    features = augmentation_instance.generate_features()
    target_mae_decrease = augmentation_instance.compute_decrease_in_mean_absolute_error()
    target_mse_decrease = augmentation_instance.compute_decrease_in_mean_squared_error()
    target_med_ae_decrease = augmentation_instance.compute_decrease_in_median_absolute_error()
    target_r2_gain = augmentation_instance.compute_gain_in_r2_score()
    r2_score_before, r2_score_after = augmentation_instance.get_before_and_after_r2_scores()
    # query and candidate information
    query_dataset = augmentation_instance.get_query_filename()
    candidate_dataset = augmentation_instance.get_candidate_filename()
    target = augmentation_instance.get_target_name()

    return [[query_dataset, target, candidate_dataset] + list(features) + [target_mae_decrease, target_mse_decrease, target_med_ae_decrease, target_r2_gain, r2_score_before, r2_score_after]]


def feature_array_to_string(feature_array):
    """Transforms feature array into a single string.
    """

    feature_array_str = [str(x) for x in feature_array]
    feature_array_str[0] = "\"" + feature_array_str[0] + "\""  # query
    feature_array_str[1] = "\"" + feature_array_str[1] + "\""  # target
    feature_array_str[2] = "\"" + feature_array_str[2] + "\""  # candidate

    return ','.join(feature_array_str)


def add_data_to_json(json_obj, query_data, candidate_data):
    """Adds query and candidate datasets to json object.
    """

    json_obj['query_data'] = query_data
    json_obj['candidate_data'] = candidate_data
    return json_obj


if __name__ == '__main__':

    start_time = time.time()

    # Spark context
    conf = SparkConf().setAppName('Feature Generation')
    sc = SparkContext(conf=conf)

    # parameters
    params = json.load(open('.params_feature_generation.json'))
    learning_data_filename = params['learning_data_filename']
    id_to_dataset_file = params['file_mapping']
    augmentation_learning_data_filename = params['augmentation_learning_data_filename']
    cluster_execution = params['cluster']
    hdfs_address = params['hdfs_address']
    hdfs_user = params['hdfs_user']

    # opening training data file
    learning_data_filename_for_spark = learning_data_filename
    if not cluster_execution:
        learning_data_filename_for_spark = 'file://' + learning_data_filename
    # line below works for 2019-10-28 data
    #learning_data = sc.textFile(learning_data_filename_for_spark).repartition(NUMBER_OF_SPARK_REPARTITIONS)
    # lne below works for 2019-11-08 data
    learning_data = sc.textFile(learning_data_filename_for_spark + '/*').persist(StorageLevel.MEMORY_AND_DISK)

    # opening id to dataset mapping
    #   format is the following:
    #   (dataset id, dataset)
    if not cluster_execution:
        id_to_dataset_file = 'file://' + id_to_dataset_file
    id_to_dataset = sc.pickleFile(id_to_dataset_file).persist(StorageLevel.MEMORY_AND_DISK)

    # generating learning instances
    learning_instances = learning_data.map(
        lambda x: json.loads(x)
    ).map(
        # first, let's use query dataset id as key
        # (query dataset id, (candidate dataset id, dict))
        lambda x: (x['query_dataset'], (x['candidate_dataset'], x))
    ).join(
        # we get the query datasets
        dataset_id_to_data
    ).map(
        # (candidate dataset id, (query dataset, dict))
        lambda x: (x[1][0][0], (x[1][1], x[1][0][1]))
    ).join(
        # we get the candidate datasets
        dataset_id_to_data
    ).map(
        lambda x: add_data_to_json(x[1][0][1], x[1][0][0], x[1][1])
    ).flatMap(
        lambda x: generate_learning_instance(x)
    ).map(
        lambda x: feature_array_to_string(x)
    )

    #learning_instances.saveAsTextFile(augmentation_learning_data_filename)

    # creating a hdfs client for reading purposes
    hdfs_client = InsecureClient(hdfs_address, user=hdfs_user)
    
    save_file(
        augmentation_learning_data_filename,
        '\n'.join(learning_instances.collect()),
        hdfs_client, 
        cluster_execution,
        hdfs_address,
        hdfs_user
    )

    print('Duration: %.4f seconds' % (time.time() - start_time))
