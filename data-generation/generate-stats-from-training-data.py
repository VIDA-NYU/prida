from hdfs import InsecureClient
from io import StringIO
import json
import os
import pandas as pd
from pyspark import SparkConf, SparkContext, StorageLevel
import sys


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


def generate_stats_from_record(record, load_dataframes, params):
    """Computes some statistics related to the training data record.
    """

    global n_records
    global before_lte_after
    global before_gt_after
    global query_size_lte_candidate_size
    global query_size_gt_candidate_size

    query = record[0]
    target = record[1]
    candidate = record[2]
    score_before = record[3]  # mean absolute error
    score_after = record[4]  # mean absolute error

    # parameters
    output_dir = params['new_datasets_directory']
    cluster_execution = params['cluster']
    hdfs_address = params['hdfs_address']
    hdfs_user = params['hdfs_user']

    # incrementing number of records
    n_records += 1

    # learning scores
    if score_before <= score_after:
        before_lte_after += 1
    else:
        before_gt_after += 1

    # dataframes
    if load_dataframes:

        query_data = pd.read_csv(StringIO(
            read_file(
                os.path.join(output_dir, 'files', query),
                cluster_execution,
                hdfs_address,
                hdfs_user)
        ))
        candidate_data = pd.read_csv(StringIO(
            read_file(
                os.path.join(output_dir, 'files', candidate),
                cluster_execution,
                hdfs_address,
                hdfs_user)
        ))

        # dataframe sizes
        if query_data.shape[0] <= candidate_data.shape[0]:
            query_size_lte_candidate_size += 1
        else:
            query_size_gt_candidate_size += 1

        # join
        query_data.set_index(
            'key-for-ranking',
            drop=True,
            inplace=True
        )
        candidate_data.set_index(
            'key-for-ranking',
            drop=True,
            inplace=True
        )

    return None


def list_dir(file_path, use_hdfs=False, hdfs_address=None, hdfs_user=None):
    """Lists all the files inside the directory specified by file_path.
    """

    if use_hdfs:
        hdfs_client = InsecureClient(hdfs_address, user=hdfs_user)
        return hdfs_client.list(file_path)
    return os.listdir(file_path)
    

if __name__ == '__main__':

    # Spark context
    conf = SparkConf().setAppName("Data Generation Stats")
    sc = SparkContext(conf=conf)

    # accumulators
    query_size_lte_candidate_size = sc.accumulator(0)
    query_size_gt_candidate_size = sc.accumulator(0)

    # parameters
    params = json.load(open(".params.json"))
    output_dir = params['new_datasets_directory']
    cluster_execution = params['cluster']
    hdfs_address = params['hdfs_address']
    hdfs_user = params['hdfs_user']

    # searching for training data
    algorithms = dict()
    load_dataframes = True
    for file_ in list_dir(output_dir, cluster_execution, hdfs_address, hdfs_user):
        if 'training-data' not in file_:
            continue
        algorithm_name = ' '.join(file_.replace('training-data-', '').split('-'))
        algorithms[algorithm_name] = dict(
            n_records=0,
            before_lte_after=0,
            before_gt_after=0
        )
        filename = os.path.join(output_dir, file_)
        if not cluster_execution:
            filename = 'file://' + filename

        # accumulators
        n_records = sc.accumulator(0)
        before_lte_after = sc.accumulator(0)
        before_gt_after = sc.accumulator(0)

        training_data = sc.textFile(filename).map(
            lambda x: generate_stats_from_record(x.split(','), load_dataframes, params)
        ).collect()

        algorithms[algorithm_name]['n_records'] = n_records.value
        algorithms[algorithm_name]['before_lte_after'] = before_lte_after.value
        algorithms[algorithm_name]['before_gt_after'] = before_gt_after.value

        load_dataframes = False

    print('')
    for algorithm in algorithms:
        print('Statistics for %s:' % algorithm)
        print(' -- Number of records: %d' % algorithms[algorithm]['n_records'])
        print(' -- MAE before lte MAE after: %d' % algorithms[algorithm]['before_lte_after'])
        print(' -- MAE before gt MAE after: %d' % algorithms[algorithm]['before_gt_after'])
        print('')

    print('General statistics:')
    print(' -- Size query lte size candidate: %s' % query_size_lte_candidate_size.value)
    print(' -- Size query gt size candidate: %s' % query_size_gt_candidate_size.value)
    print('')
