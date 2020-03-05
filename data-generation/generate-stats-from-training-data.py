from hdfs import InsecureClient
from io import StringIO
import json
import numpy as np
from operator import add
import os
import pandas as pd
from pyspark import SparkConf, SparkContext, StorageLevel
import sys


def list_dir(file_path, hdfs_client=None, use_hdfs=False):
    """Lists all the files inside the directory specified by file_path.
    """

    if use_hdfs:
        return hdfs_client.list(file_path)
    return os.listdir(file_path)


def generate_stats_from_record(record, load_dataframes):
    """Computes some statistics related to the training data record.
    """

    global n_records
    global before_mae_lte_after
    global before_mae_gt_after
    global before_mse_lte_after
    global before_mse_gt_after
    global before_mdae_lte_after
    global before_mdae_gt_after
    global before_r2_lte_after
    global before_r2_gt_after
    global query_size_lte_candidate_size
    global query_size_gt_candidate_size

    query = record['query_dataset']
    query_data_obj = record['query_data']
    target = record['target']
    candidate = record['candidate_dataset']
    candidate_data_obj = record['candidate_data']
    imputation_strategy = record['imputation_strategy']
    mae_before = record['mean_absolute_error'][0]
    mae_after = record['mean_absolute_error'][1]
    mse_before = record['mean_squared_error'][0]
    mse_after = record['mean_squared_error'][1]
    mdae_before = record['median_absolute_error'][0]
    mdae_after = record['median_absolute_error'][1]
    r2_before = record['r2_score'][0]
    r2_after = record['r2_score'][1]

    # incrementing number of records
    n_records += 1

    # learning scores
    if mae_before <= mae_after:
        before_mae_lte_after += 1
    else:
        before_mae_gt_after += 1
    if mse_before <= mse_after:
        before_mse_lte_after += 1
    else:
        before_mse_gt_after += 1
    if mdae_before <= mdae_after:
        before_mdae_lte_after += 1
    else:
        before_mdae_gt_after += 1
    if r2_before <= r2_after:
        before_r2_lte_after += 1
    else:
        before_r2_gt_after += 1

    # dataframes
    if load_dataframes:

        query_data = pd.read_csv(StringIO(query_data_obj))
        candidate_data = pd.read_csv(StringIO(candidate_data_obj))

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
                query_data.shape[0], query_data.shape[1], candidate_data.shape[0], candidate_data.shape[1],
                len(query_data_obj.encode('utf-8')), len(candidate_data_obj.encode('utf-8')))

    return (imputation_strategy, None, None, None, None, None, None, None, None)


def add_data_to_json(json_obj, query_data, candidate_data):
    """Adds query and candidate datasets to json object.
    """

    json_obj['query_data'] = query_data
    json_obj['candidate_data'] = candidate_data
    return json_obj
    

if __name__ == '__main__':

    # Spark context
    conf = SparkConf().setAppName("Data Generation Stats")
    sc = SparkContext(conf=conf)

    # accumulators and global variables
    query_size_lte_candidate_size = sc.accumulator(0)
    query_size_gt_candidate_size = sc.accumulator(0)
    query_intersection_sizes = list()
    candidate_intersection_sizes = list()
    query_candidate_size = list()
    query_n_rows = list()
    query_n_columns = list()
    candidate_n_rows = list()
    candidate_n_columns = list()
    query_size_bytes = list()
    candidate_size_bytes = list()

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

    # dataset mappings
    id_to_dataset_filename_training = os.path.join(output_dir, 'id-to-dataset-training')
    id_to_dataset_filename_test = os.path.join(output_dir, 'id-to-dataset-test')
    if not cluster_execution:
        id_to_dataset_filename_training = 'file://' + id_to_dataset_filename_training
        id_to_dataset_filename_test = 'file://' + id_to_dataset_filename_test

    id_to_dataset = dict()
    id_to_dataset['training'] = sc.pickleFile(
        id_to_dataset_filename_training
    ).persist(StorageLevel.MEMORY_AND_DISK)
    id_to_dataset['test'] = sc.pickleFile(
        id_to_dataset_filename_test
    ).persist(StorageLevel.MEMORY_AND_DISK)

    # searching for training data
    algorithms = dict()
    load_dataframes = True
    for file_ in list_dir(output_dir, hdfs_client, cluster_execution):
        for key in ['training', 'test']:
            if '%s-data-'%key not in file_:
                continue
            algorithm_name = ' '.join(file_.replace('%s-data-'%key, '').split('-'))
            if algorithm_name not in algorithms:
                algorithms[algorithm_name] = dict(
                    n_records=0,
                    before_lte_after=0,
                    before_gt_after=0,
                    imputation_strategies=dict()
                )
            filename = os.path.join(output_dir, file_ + '/*')
            if not cluster_execution:
                filename = 'file://' + filename

            # accumulators
            n_records = sc.accumulator(0)
            before_mae_lte_after = sc.accumulator(0)
            before_mae_gt_after = sc.accumulator(0)
            before_mse_lte_after = sc.accumulator(0)
            before_mse_gt_after = sc.accumulator(0)
            before_mdae_lte_after = sc.accumulator(0)
            before_mdae_gt_after = sc.accumulator(0)
            before_r2_lte_after = sc.accumulator(0)
            before_r2_gt_after = sc.accumulator(0)

            stats = sc.emptyRDD()
            if load_dataframes:
                stats = sc.textFile(filename).map(
                    lambda x: json.loads(x)
                ).map(
                    # first, let's use query dataset id as key
                    # (query dataset id, (candidate dataset id, dict))
                    lambda x: (x['query_dataset'], (x['candidate_dataset'], x))
                ).join(
                    # we get the query datasets
                    id_to_dataset[key]
                ).map(
                    # (candidate dataset id, (query dataset, dict))
                    lambda x: (x[1][0][0], (x[1][1], x[1][0][1]))
                ).join(
                    # we get the candidate datasets
                    id_to_dataset[key]
                ).repartition(372).map(
                    lambda x: add_data_to_json(x[1][0][1], x[1][0][0], x[1][1])
                ).map(
                    lambda x: generate_stats_from_record(x, load_dataframes)
                ).persist(StorageLevel.MEMORY_AND_DISK)
            else:
                stats = sc.textFile(filename).repartition(372).map(
                    lambda x: add_data_to_json(json.loads(x), None, None)
                ).map(
                    lambda x: generate_stats_from_record(x, load_dataframes)
                ).persist(StorageLevel.MEMORY_AND_DISK)

            imputation_strategies = stats.map(
                lambda x: (x[0], 1)
            ).reduceByKey(add).collect()

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

            size_bytes = stats.filter(
                lambda x: x[1] != None and x[2] != None
            ).map(
                lambda x: (x[7], x[8])
            ).collect()

            if len(intersection_sizes) > 0:
                query_intersection_sizes += [x for (x, y) in intersection_sizes]
                candidate_intersection_sizes += [y for (x, y) in intersection_sizes]

            if len(n_rows_columns) > 0:
                query_n_rows += [x for (x, y, w, z) in n_rows_columns]
                query_n_columns += [y for (x, y, w, z) in n_rows_columns]
                candidate_n_rows += [w for (x, y, w, z) in n_rows_columns]
                candidate_n_columns += [z for (x, y, w, z) in n_rows_columns]
                query_candidate_size += [y + z - 1 for (x, y, w, z) in n_rows_columns]

            if len(size_bytes) > 0:
                query_size_bytes += [x for (x, y) in size_bytes]
                candidate_size_bytes += [y for (x, y) in size_bytes]

            algorithms[algorithm_name]['n_records'] += n_records.value
            algorithms[algorithm_name]['before_mae_lte_after'] += before_mae_lte_after.value
            algorithms[algorithm_name]['before_mae_gt_after'] += before_mae_gt_after.value
            algorithms[algorithm_name]['before_mse_lte_after'] += before_mse_lte_after.value
            algorithms[algorithm_name]['before_mse_gt_after'] += before_mse_gt_after.value
            algorithms[algorithm_name]['before_mdae_lte_after'] += before_mdae_lte_after.value
            algorithms[algorithm_name]['before_mdae_gt_after'] += before_mdae_gt_after.value
            algorithms[algorithm_name]['before_r2_lte_after'] += before_r2_lte_after.value
            algorithms[algorithm_name]['before_r2_gt_after'] += before_r2_gt_after.value
            for (k, v) in imputation_strategies:
                if k not in algorithms[algorithm_name]['imputation_strategies']:
                    algorithms[algorithm_name]['imputation_strategies'][k] = 0
                algorithms[algorithm_name]['imputation_strategies'][k] += v

            load_dataframes = False

    print('')
    for algorithm in algorithms:
        print('Statistics for %s:' % algorithm)
        print(' -- Number of records: %d' % algorithms[algorithm]['n_records'])
        print(' -- MAE before gt MAE after: %d (%.2f%%)' % (
            algorithms[algorithm]['before_mae_gt_after'],
            (100 * algorithms[algorithm]['before_mae_gt_after']) / algorithms[algorithm]['n_records']
        ))
        print(' -- MAE before lte MAE after: %d (%.2f%%)' % (
            algorithms[algorithm]['before_mae_lte_after'],
            (100 * algorithms[algorithm]['before_mae_lte_after']) / algorithms[algorithm]['n_records']
        ))
        print(' -- MSE before gt MSE after: %d (%.2f%%)' % (
            algorithms[algorithm]['before_mse_gt_after'],
            (100 * algorithms[algorithm]['before_mse_gt_after']) / algorithms[algorithm]['n_records']
        ))
        print(' -- MSE before lte MSE after: %d (%.2f%%)' % (
            algorithms[algorithm]['before_mse_lte_after'],
            (100 * algorithms[algorithm]['before_mse_lte_after']) / algorithms[algorithm]['n_records']
        ))
        print(' -- MDAE before gt MDAE after: %d (%.2f%%)' % (
            algorithms[algorithm]['before_mdae_gt_after'],
            (100 * algorithms[algorithm]['before_mdae_gt_after']) / algorithms[algorithm]['n_records']
        ))
        print(' -- MDAE before lte MDAE after: %d (%.2f%%)' % (
            algorithms[algorithm]['before_mdae_lte_after'],
            (100 * algorithms[algorithm]['before_mdae_lte_after']) / algorithms[algorithm]['n_records']
        ))
        print(' -- R^2 before gt R^2 after: %d (%.2f%%)' % (
            algorithms[algorithm]['before_r2_gt_after'],
            (100 * algorithms[algorithm]['before_r2_gt_after']) / algorithms[algorithm]['n_records']
        ))
        print(' -- R^2 before lte R^2 after: %d (%.2f%%)' % (
            algorithms[algorithm]['before_r2_lte_after'],
            (100 * algorithms[algorithm]['before_r2_lte_after']) / algorithms[algorithm]['n_records']
        ))
        print(' -- Missing value imputation strategies:')
        for (strategy, count) in sorted(algorithms[algorithm]['imputation_strategies'], key=lambda x: x[1], reverse=True):
            print('    . %s\t%d' % (strategy, count))
        print('')

    hist_query_intersection_size = np.histogram(query_intersection_sizes, bins=10)
    hist_candidate_intersection_size = np.histogram(candidate_intersection_sizes, bins=10)

    hist_query_n_rows = np.histogram(query_n_rows, bins=10)
    hist_query_n_columns = np.histogram(query_n_columns, bins=10)
    hist_candidate_n_rows = np.histogram(candidate_n_rows, bins=10)
    hist_candidate_n_columns = np.histogram(candidate_n_columns, bins=10)
    hist_query_candidate_size = np.histogram(query_candidate_size, bins=10)

    hist_query_size_bytes = np.histogram(query_size_bytes, bins=10)
    hist_candidate_size_bytes = np.histogram(candidate_size_bytes, bins=10)

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
    print(' -- Join size (number of columns): ')
    for i in range(1, len(hist_query_candidate_size[1])):
        print('    [%.4f, %4f]\t%d' % (
            hist_query_candidate_size[1][i-1],
            hist_query_candidate_size[1][i],
            hist_query_candidate_size[0][i-1])
        )
    print(' -- Query size (bytes): ')
    for i in range(1, len(hist_query_size_bytes[1])):
        print('    [%.4f, %4f]\t%d' % (
            hist_query_size_bytes[1][i-1],
            hist_query_size_bytes[1][i],
            hist_query_size_bytes[0][i-1])
        )
    print(' -- Candidate size (bytes): ')
    for i in range(1, len(hist_candidate_size_bytes[1])):
        print('    [%.4f, %4f]\t%d' % (
            hist_candidate_size_bytes[1][i-1],
            hist_candidate_size_bytes[1][i],
            hist_candidate_size_bytes[0][i-1])
        )
    print('')

    print('Configuration:')
    print(' -- new_datasets_directory: %s' % params['new_datasets_directory'])
    print(' -- cluster: %s' % str(params['cluster']))
    print(' -- hdfs_address: %s' % params['hdfs_address'])
    print(' -- hdfs_user: %s' % params['hdfs_user'])
    print('')
