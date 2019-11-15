from hdfs import InsecureClient
from io import StringIO
import json
import numpy as np
import os
import pandas as pd
from pyspark import SparkConf, SparkContext, StorageLevel
import sys


def read_file(file_path, hdfs_client=None, use_hdfs=False):
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


def list_dir(file_path, hdfs_client=None, use_hdfs=False):
    """Lists all the files inside the directory specified by file_path.
    """

    if use_hdfs:
        return hdfs_client.list(file_path)
    return os.listdir(file_path)


def generate_stats_from_dataset(dataset, params):
    """Computes some statistics related to the dataset.
    """

    global dataframe_exception
    global processed_datasets

    # File system information
    cluster_execution = params['cluster']
    hdfs_address = params['hdfs_address']
    hdfs_user = params['hdfs_user']

    # HDFS Client
    hdfs_client = None
    if cluster_execution:
        hdfs_client = InsecureClient(hdfs_address, user=hdfs_user)

    data = None
    try:
        data = pd.read_csv(StringIO(
            read_file(
                dataset,
                hdfs_client,
                cluster_execution)
        ))
    except Exception as e:
        print('[WARNING] The following dataset had an exception while parsing into a dataframe: %s (%s)' % (dataset, str(e)))
        dataframe_exception += 1
        return []

    processed_datasets += 1
    return [(data.shape[0], data.shape[1])]
    

if __name__ == '__main__':

    # Spark context
    conf = SparkConf().setAppName("OpenML Stats")
    sc = SparkContext(conf=conf)

    # global variables and accumulators
    n_rows = list()
    n_columns = list()
    dataframe_exception = sc.accumulator(0)
    processed_datasets = sc.accumulator(0)

    # parameters
    params = json.load(open(".params.json"))
    output_dir = params['original_datasets_directory']
    cluster_execution = params['cluster']
    hdfs_address = params['hdfs_address']
    hdfs_user = params['hdfs_user']

    # HDFS Client
    hdfs_client = None
    if cluster_execution:
        hdfs_client = InsecureClient(hdfs_address, user=hdfs_user)

    # searching for data
    dataset_files = list()
    for dataset_path in list_dir(output_dir, hdfs_client, cluster_execution):
        for f in list_dir(os.path.join(output_dir, dataset_path), hdfs_client, cluster_execution):
            if 'learningData.csv' in f:
                dataset_files.append(os.path.join(output_dir, dataset_path, f))

    # computing stats
    files = sc.parallelize(dataset_files, 365)
    stats = files.flatMap(
        lambda x: generate_stats_from_dataset(x, params)
    ).persist(StorageLevel.MEMORY_AND_DISK)

    n_rows = stats.map(
        lambda x: x[0]
    ).collect()

    n_columns = stats.map(
        lambda x: x[1]
    ).collect()

    hist_n_rows = np.histogram(n_rows, bins=10)
    hist_n_columns = np.histogram(n_columns, bins=10)

    print(' -- N. Rows:')
    for i in range(1, len(hist_n_rows[1])):
        print('    [%.4f, %4f]\t%d' % (
            hist_n_rows[1][i-1],
            hist_n_rows[1][i],
            hist_n_rows[0][i-1])
        )
    print(' -- N. Columns:')
    for i in range(1, len(hist_n_columns[1])):
        print('    [%.4f, %4f]\t%d' % (
            hist_n_columns[1][i-1],
            hist_n_columns[1][i],
            hist_n_columns[0][i-1])
        )
    print('')

    print('Stats:')
    print(' -- Processed datasets: %d' %processed_datasets.value)
    print(' -- Datasets w/ pandas.Dataframe exception: %d' %dataframe_exception.value)
    print('')

    print('Configuration:')
    print(' -- original_datasets_directory: %s' % params['original_datasets_directory'])
    print(' -- cluster: %s' % str(params['cluster']))
    print(' -- hdfs_address: %s' % params['hdfs_address'])
    print(' -- hdfs_user: %s' % params['hdfs_user'])
    print('')
