from hdfs import InsecureClient
from io import StringIO
import json
import numpy as np
from operator import add
import os
import pandas as pd
from pyspark import SparkConf, SparkContext, StorageLevel
import sys


def read_file(file_path, hdfs_client=None, use_hdfs=False, hdfs_address=None, hdfs_user=None):
    """Opens a file for read and returns its corresponding content.
    """

    output = None
    if use_hdfs:
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

    # File system information
    cluster_execution = params['cluster']
    hdfs_address = params['hdfs_address']
    hdfs_user = params['hdfs_user']

    # HDFS Client
    hdfs_client = None
    if cluster_execution:
        hdfs_client = InsecureClient(hdfs_address, user=hdfs_user)

    query = record['query_dataset']
    target = record['target']
    candidate = record['candidate_dataset']
    imputation_strategy = record['imputation_strategy']
    score_before = record['mean_absolute_error'][0]
    score_after = record['mean_absolute_error'][1]

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
                hdfs_client,
                cluster_execution,
                hdfs_address,
                hdfs_user)
        ))
        candidate_data = pd.read_csv(StringIO(
            read_file(
                os.path.join(output_dir, 'files', candidate),
                hdfs_client,
                cluster_execution,
                hdfs_address,
                hdfs_user)
        ))

        # dataframe sizes
        if query_data.shape[0] <= candidate_data.shape[0]:
            query_size_lte_candidate_size += 1
        else:
            query_size_gt_candidate_size += 1

        # keys
        query_data_keys = set(query_data['key-for-ranking'])
        candidate_data_keys = set(candidate_data['key-for-ranking'])

        # relative intersection size
        intersection_size = len(query_data_keys & candidate_data_keys)
        query_intersection_size = intersection_size / len(query_data_keys)
        candidate_intersection_size = intersection_size / len(candidate_data_keys)

        return (imputation_strategy, query_intersection_size, candidate_intersection_size,
                query_data.shape[0], query_data.shape[1], candidate_data.shape[0], candidate_data.shape[1])

    return (imputation_strategy, None, None, None, None, None, None)


def list_dir(file_path, hdfs_client=None, use_hdfs=False, hdfs_address=None, hdfs_user=None):
    """Lists all the files inside the directory specified by file_path.
    """

    if use_hdfs:
        return hdfs_client.list(file_path)
    return os.listdir(file_path)
    

if __name__ == '__main__':

    # Spark context
    conf = SparkConf().setAppName("Data Generation Stats")
    sc = SparkContext(conf=conf)

    # accumulators and global variables
    query_size_lte_candidate_size = sc.accumulator(0)
    query_size_gt_candidate_size = sc.accumulator(0)
    query_intersection_sizes = list()
    candidate_intersection_sizes = list()
    query_n_rows = list()
    query_n_columns = list()
    candidate_n_rows = list()
    candidate_n_columns = list()

    # parameters
    params = json.load(open(".params.json"))
    output_dir = params['new_datasets_directory']
    cluster_execution = params['cluster']
    hdfs_address = params['hdfs_address']
    hdfs_user = params['hdfs_user']

    # HDFS Client
    hdfs_client = None
    if cluster_execution:
        hdfs_client = InsecureClient(hdfs_address, user=hdfs_user)

    # searching for training data
    algorithms = dict()
    load_dataframes = True
    for file_ in list_dir(output_dir, hdfs_client, cluster_execution, hdfs_address, hdfs_user):
        if 'training-data-' not in file_:
            continue
        algorithm_name = ' '.join(file_.replace('training-data-', '').split('-'))
        algorithms[algorithm_name] = dict(
            n_records=0,
            before_lte_after=0,
            before_gt_after=0,
            imputation_strategies=list()
        )
        filename = os.path.join(output_dir, file_ + '/*')
        if not cluster_execution:
            filename = 'file://' + filename

        # accumulators
        n_records = sc.accumulator(0)
        before_lte_after = sc.accumulator(0)
        before_gt_after = sc.accumulator(0)

        stats = sc.textFile(filename).map(
            lambda x: generate_stats_from_record(json.loads(x), load_dataframes, params)
        ).persist(StorageLevel.MEMORY_AND_DISK)

        imputation_strategies = sorted(stats.map(
            lambda x: (x[0], 1)
        ).reduceByKey(add).collect(), key=lambda x: x[1], reverse=True)

        intersection_sizes = stats.filter(
            lambda x: x[1] != None and x[2] != None
        ).map(
            lambda x: (x[1], x[2])
        ).collect()

        n_rows_columns = stats.filter(
            lambda x: x[1] != None and x[2] != None
        ).map(
            lambda x: (x[3], x[4], x[5], x[6])
        ).collect()

        if len(intersection_sizes) > 0:
            query_intersection_sizes += [x for (x, y) in intersection_sizes]
            candidate_intersection_sizes += [y for (x, y) in intersection_sizes]

        if len(n_rows_columns) > 0:
            query_n_rows += [x for (x, y, w, z) in n_rows_columns]
            query_n_columns += [y for (x, y, w, z) in n_rows_columns]
            candidate_n_rows += [w for (x, y, w, z) in n_rows_columns]
            candidate_n_columns += [z for (x, y, w, z) in n_rows_columns]

        algorithms[algorithm_name]['n_records'] = n_records.value
        algorithms[algorithm_name]['before_lte_after'] = before_lte_after.value
        algorithms[algorithm_name]['before_gt_after'] = before_gt_after.value
        algorithms[algorithm_name]['imputation_strategies'] = imputation_strategies

        load_dataframes = False

    print('')
    for algorithm in algorithms:
        print('Statistics for %s:' % algorithm)
        print(' -- Number of records: %d' % algorithms[algorithm]['n_records'])
        print(' -- MAE before gt MAE after: %d (%.2f%%)' % (
            algorithms[algorithm]['before_gt_after'],
            (100 * algorithms[algorithm]['before_gt_after']) / algorithms[algorithm]['n_records']
        ))
        print(' -- MAE before lte MAE after: %d (%.2f%%)' % (
            algorithms[algorithm]['before_lte_after'],
            (100 * algorithms[algorithm]['before_lte_after']) / algorithms[algorithm]['n_records']
        ))
        print(' -- Missing value imputation strategies:')
        for (strategy, count) in algorithms[algorithm]['imputation_strategies']:
            print('    . %s\t%d' % (strategy, count))
        print('')

    hist_query_intersection_size = np.histogram(query_intersection_sizes, bins=10)
    hist_candidate_intersection_size = np.histogram(candidate_intersection_sizes, bins=10)

    hist_query_n_rows = np.histogram(query_n_rows, bins=10)
    hist_query_n_columns = np.histogram(query_n_columns, bins=10)
    hist_candidate_n_rows = np.histogram(candidate_n_rows, bins=10)
    hist_candidate_n_columns = np.histogram(candidate_n_columns, bins=10)

    print('General statistics:')
    print(' -- Size query lte size candidate: %d (%.2f%%)' % (
        query_size_lte_candidate_size.value,
        (100 * query_size_lte_candidate_size.value) / (query_size_lte_candidate_size.value + query_size_gt_candidate_size.value)
    ))
    print(' -- Size query gt size candidate: %d (%.2f%%)' % (
        query_size_gt_candidate_size.value,
        (100 * query_size_gt_candidate_size.value) / (query_size_lte_candidate_size.value + query_size_gt_candidate_size.value)
    ))
    print(' -- Query intersection size: ')
    for i in range(1, len(hist_query_intersection_size[1])):
        print('    [%.4f, %4f]\t%d' % (
            hist_query_intersection_size[1][i-1],
            hist_query_intersection_size[1][i],
            hist_query_intersection_size[0][i-1])
        )
    print(' -- Candidate intersection size: ')
    for i in range(1, len(hist_candidate_intersection_size[1])):
        print('    [%.4f, %4f]\t%d' % (
            hist_candidate_intersection_size[1][i-1],
            hist_candidate_intersection_size[1][i],
            hist_candidate_intersection_size[0][i-1])
        )
    print(' -- Query number of records: ')
    for i in range(1, len(hist_query_n_rows[1])):
        print('    [%.4f, %4f]\t%d' % (
            hist_query_n_rows[1][i-1],
            hist_query_n_rows[1][i],
            hist_query_n_rows[0][i-1])
        )
    print(' -- Query number of columns: ')
    for i in range(1, len(hist_query_n_columns[1])):
        print('    [%.4f, %4f]\t%d' % (
            hist_query_n_columns[1][i-1],
            hist_query_n_columns[1][i],
            hist_query_n_columns[0][i-1])
        )
    print(' -- Candidate number of records: ')
    for i in range(1, len(hist_candidate_n_rows[1])):
        print('    [%.4f, %4f]\t%d' % (
            hist_candidate_n_rows[1][i-1],
            hist_candidate_n_rows[1][i],
            hist_candidate_n_rows[0][i-1])
        )
    print(' -- Candidate number of columns: ')
    for i in range(1, len(hist_candidate_n_columns[1])):
        print('    [%.4f, %4f]\t%d' % (
            hist_candidate_n_columns[1][i-1],
            hist_candidate_n_columns[1][i],
            hist_candidate_n_columns[0][i-1])
        )
    print('')
