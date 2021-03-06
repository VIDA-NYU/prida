#!/usr/bin/env python3
from augmentation_instance import *
import json
from pyspark import SparkConf, SparkContext, StorageLevel
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
    mark = augmentation_instance.get_mark()

    return [[query_dataset, target, candidate_dataset, mark] + list(features) + [target_mae_decrease, target_mse_decrease, target_med_ae_decrease, target_r2_gain, r2_score_before, r2_score_after]]


def feature_array_to_string(feature_array):
    """Transforms feature array into a single string.
    """

    feature_array_str = [str(x) for x in feature_array]
    feature_array_str[0] = "\"" + feature_array_str[0] + "\""  # query
    feature_array_str[1] = "\"" + feature_array_str[1] + "\""  # target
    feature_array_str[2] = "\"" + feature_array_str[2] + "\""  # candidate
    feature_array_str[3] = "\"" + feature_array_str[3] + "\""  # mark

    return ','.join(feature_array_str)


def add_data_to_json(json_obj, query_data, candidate_data, joined_data=None):
    """Adds query and candidate datasets to json object.
    """

    json_obj['query_data'] = query_data
    json_obj['candidate_data'] = candidate_data
    if joined_data:
        json_obj['joined_data'] = joined_data
    return json_obj


def generate_learning_instances(learning_data, dataset_id_to_data):
    """Preprocess the data and generates features from the records.
    """

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
    ).repartition(1000).map(
        lambda x: add_data_to_json(x[1][0][1], x[1][0][0], x[1][1])
    ).flatMap(
        lambda x: generate_learning_instance(x)
    ).map(
        lambda x: feature_array_to_string(x)
    )

    return learning_instances


def generate_learning_instances_with_joined_dataset(learning_data, dataset_id_to_data):
    """Preprocess the data and generates features from the records.
    """

    learning_instances = learning_data.map(
        lambda x: json.loads(x)
    ).map(
        # first, let's use query dataset id as key
        # (query dataset id, (candidate dataset id, joined dataset id, dict))
        lambda x: (x['query_dataset'], (x['candidate_dataset'], x['joined_dataset'], x))
    ).join(
        # we get the query datasets
        dataset_id_to_data
    ).map(
        # (candidate dataset id, (query dataset, joined dataset id, dict))
        lambda x: (x[1][0][0], (x[1][1], x[1][0][1], x[1][0][2]))
    ).join(
        # we get the candidate datasets
        dataset_id_to_data
    ).map(
        # (joined dataset id, (query dataset, candidate dataset, dict))
        lambda x: (x[1][0][1], (x[1][0][0], x[1][1], x[1][0][2]))
    ).join(
        # we get the joined datasets
        dataset_id_to_data
    ).repartition(1000).map(
        lambda x: add_data_to_json(x[1][0][2], x[1][0][0], x[1][0][1], x[1][1])
    ).flatMap(
        lambda x: generate_learning_instance(x)
    ).map(
        lambda x: feature_array_to_string(x)
    )

    return learning_instances


if __name__ == '__main__':

    start_time = time.time()

    # Spark context
    conf = SparkConf().setAppName('Feature Generation')
    sc = SparkContext(conf=conf)

    # parameters
    params = json.load(open('.params_feature_generation.json'))
    learning_data_filename_training = params['learning_data_training']
    learning_data_filename_test = None if params['learning_data_test'] == '' else params['learning_data_test']
    id_to_dataset_filename_training = params['id_to_dataset_training']
    id_to_dataset_filename_test = None if params['id_to_dataset_test'] == '' else params['id_to_dataset_test']
    augmentation_learning_data_filename = params['augmentation_learning_data_filename']
    cluster_execution = params['cluster']
    hdfs_address = params['hdfs_address']
    hdfs_user = params['hdfs_user']

    # creating a hdfs client for writing purposes
    hdfs_client = InsecureClient(hdfs_address, user=hdfs_user)

    # opening training and test data files
    if not cluster_execution:
        learning_data_filename_training = 'file://' + learning_data_filename_training
        id_to_dataset_filename_training = 'file://' + id_to_dataset_filename_training
        if learning_data_filename_test:
            learning_data_filename_test = 'file://' + learning_data_filename_test
            id_to_dataset_filename_test = 'file://' + id_to_dataset_filename_test
    
    learning_data_training = sc.textFile(learning_data_filename_training + '/*').persist(StorageLevel.MEMORY_AND_DISK)
    id_to_dataset_training = sc.pickleFile(id_to_dataset_filename_training).persist(StorageLevel.MEMORY_AND_DISK)
    learning_data_test = sc.emptyRDD()
    id_to_dataset_test = sc.emptyRDD()
    if learning_data_filename_test:
        learning_data_test = sc.textFile(learning_data_filename_test + '/*').persist(StorageLevel.MEMORY_AND_DISK)
        id_to_dataset_test = sc.pickleFile(id_to_dataset_filename_test).persist(StorageLevel.MEMORY_AND_DISK)

    # taking first element and checking if information about joined dataset is present
    has_joined_data = False
    first = json.loads(learning_data_training.first())
    if 'joined_dataset' in first:
        has_joined_data = True

    # generating learning instances for training
    if not has_joined_data:
        learning_instances_training = generate_learning_instances(
            learning_data_training,
            id_to_dataset_training
        )
    else:
        learning_instances_training = generate_learning_instances_with_joined_dataset(
            learning_data_training,
            id_to_dataset_training
        )
    
    filename = augmentation_learning_data_filename + '-training'
    delete_dir(filename, hdfs_client, cluster_execution)
    if not cluster_execution:
        filename = 'file://' + filename
    learning_instances_training.saveAsTextFile(filename)

    # freeing some memory
    learning_data_training.unpersist()
    id_to_dataset_training.unpersist()

    # generating learning instances for test
    if not learning_data_test.isEmpty():
        if not has_joined_data:
            learning_instances_test = generate_learning_instances(
                learning_data_test,
                id_to_dataset_test
            )
        else:
            learning_instances_test = generate_learning_instances_with_joined_dataset(
                learning_data_test,
                id_to_dataset_test
            )

        filename = augmentation_learning_data_filename + '-test'
        delete_dir(filename, hdfs_client, cluster_execution)
        if not cluster_execution:
            filename = 'file://' + filename
        learning_instances_test.saveAsTextFile(filename)

    print('Duration: %.4f seconds' % (time.time() - start_time))
