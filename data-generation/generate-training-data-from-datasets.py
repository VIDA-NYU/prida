#!/usr/bin/env python3
from hdfs import InsecureClient
from io import StringIO
import json
import numpy as np
import os
import pandas as pd
from pyspark import SparkConf, SparkContext, StorageLevel
import random
import shutil
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import string
import sys
import time
import uuid


def organize_dataset_files(file_path, cluster_execution):
    """Map function to organizes the dataset files from one dataset.
    """

    dataset_name = ''
    file_name = os.path.basename(os.path.normpath(file_path))
    if cluster_execution:
        # path is 'DATASET_NAME/FILE_NAME'
        dataset_name = file_path.split(os.path.sep)[-2]
    else:
        if file_name == 'learningData.csv':
            # path is 'DATASET_NAME/DATASET_NAME_dataset/tables/learningData.csv'
            dataset_name = file_path.split(os.path.sep)[-4]
        else:
            # path is 'DATASET_NAME/DATASET_NAME_dataset/datasetDoc.json' or
            #   'DATASET_NAME/DATASET_NAME_problem/problemDoc.json'
            dataset_name = file_path.split(os.path.sep)[-3]

    return (dataset_name, [(file_name, file_path)])


def read_file(file_path, use_hdfs=False, hdfs_address=None, hdfs_user=None):
    """Opens a file for read and returns its corresponding content.
    """

    output = None
    if use_hdfs:
        hdfs_client = InsecureClient(hdfs_address, user=hdfs_user)
        with hdfs_client.read(file_path) as reader:
            output = reader.read()
    else:
        with open(file_path) as reader:
            output = reader.read()
    return output


def save_file(file_path, content, use_hdfs=False, hdfs_address=None, hdfs_user=None):
    """Opens a file for write and returns its corresponding file object.
    """

    if use_hdfs:
        hdfs_client = InsecureClient(hdfs_address, user=hdfs_user)
        with hdfs_client.write(file_path) as writer:
            writer.write(content)
    else:
        with open(file_path, 'w') as writer:
            writer.write(content)


def create_dir(file_path, use_hdfs=False, hdfs_address=None, hdfs_user=None):
    """Creates a new directory specified by file_path.
    Returns True on success.
    """

    if use_hdfs:
        hdfs_client = InsecureClient(hdfs_address, user=hdfs_user)
        if hdfs_client.status(file_path, strict=False):
            hdfs_client.delete(file_path, recursive=True)
        hdfs_client.makedirs(file_path)
    else:
        if os.path.exists(file_path):
            shutil.rmtree(file_path)
        os.makedirs(file_path)
    return True


def generate_query_and_candidate_datasets_positive_examples(input_dataset, params):
    """Generates query and cadidate datasets from a single dataset
    for positive examples.

    The format of input_dataset is as follows:

      (dataset_name, [('learningData.csv', path),
                      ('datasetDoc.json', path),
                      ('problemDoc.json', path)])
    """

    result = list()

    # params
    algorithm = params['regression_algorithm']
    max_times_break_data_vertical = params['max_times_break_data_vertical']
    ignore_first_attribute = params['ignore_first_attribute']
    output_dir = params['output_directory']
    cluster_execution = params['cluster']
    hdfs_address = params['hdfs_address']
    hdfs_user = params['hdfs_user']

    problem_type = None
    target_variable = None
    multiple_data = False
    column_metadata = dict()

    data_name = input_dataset[0]
    data_file = None
    dataset_doc =  None
    problem_doc = None
    for d in input_dataset[1]:
        if d[0] == 'learningData.csv':
            data_file = read_file(d[1], cluster_execution, hdfs_address, hdfs_user)
            continue
        if d[0] == 'datasetDoc.json':
            dataset_doc = json.loads(read_file(d[1], cluster_execution, hdfs_address, hdfs_user))
            continue
        if d[0] == 'problemDoc.json':
            problem_doc = json.loads(read_file(d[1], cluster_execution, hdfs_address, hdfs_user))
    
    # problem type
    if problem_doc.get('about') and problem_doc['about'].get('taskType'):
        problem_type = problem_doc['about']['taskType']
    # target variable
    if problem_doc.get('inputs') and problem_doc['inputs'].get('data'):
        if len(problem_doc['inputs']['data']) > 0:
            data_ = problem_doc['inputs']['data'][0]
            if data_.get('targets') and len(data_['targets']) > 0:
                target_variable = data_['targets'][0]['colIndex']
    # multiple data?
    if dataset_doc.get('dataResources') and len(dataset_doc['dataResources']) > 1:
        multiple_data = True
    # column metadata
    if dataset_doc.get('dataResources') and len(dataset_doc['dataResources']) > 0:
        for metadata in dataset_doc['dataResources']:
            if metadata['resPath'] == 'tables/learningData.csv':
                for column in metadata['columns']:
                    column_metadata[column['colIndex']] = column['colType']
                break

    # regression problems only
    if problem_type != 'regression':
        print('The following dataset does not belong to a regression problem: %s (%s)' % (data_name, problem_type))
        return result
    # single data tables only
    if multiple_data:
        print('The following dataset is composed by multiple files: %s' % data_name)
        return result

    # non-numeric attributes
    n_non_numeric_att = 0
    non_numeric_att_list = list()
    for col in column_metadata:
        if col == 0 and ignore_first_attribute:
            continue
        if 'real' not in column_metadata[col] and 'integer' not in column_metadata[col]:
            n_non_numeric_att += 1
            non_numeric_att_list.append(col)

    if target_variable in non_numeric_att_list:
        print('The following dataset has a non-numerical target variable: %s' % data_name)
        return result

    # removing target variable, non-numeric attributes, and first attribute (if it is to be ignored)
    n_columns_left = len(column_metadata) - 1 - n_non_numeric_att
    if ignore_first_attribute:
        n_columns_left -= 1
    # if there is only one column left, there is no way to
    # generate both query and candidate datasets
    if n_columns_left <= 1:
        print('The following dataset does not have enough columns for the data generation process: %s' % data_name)
        return result

    # potential numbers of columns in a query dataset
    # for instance, if a dataset has 3 columns left (ignoring the target variable),
    #   a query dataset could have either 1 or 2 of these columns (so that the
    #   candidate dataset can have 1 or 2 columns, respectively)
    # in this example, n_columns_query_dataset = [1, 2]
    n_potential_columns_query_dataset = list(range(1, n_columns_left))

    # maximum number of times that the original data will be vertically broken into
    #   multiple datasets
    n_vertical_data = np.random.choice(
        list(range(1, min(max_times_break_data_vertical, n_columns_left)))
    )

    # number of columns for each time the data is vertically broken
    n_columns_query_dataset = np.random.choice(
        n_potential_columns_query_dataset,
        n_vertical_data,
        replace=False
    )

    # list of column indices
    range_start = 1 if ignore_first_attribute else 0
    all_columns = list(range(range_start, len(column_metadata)))
    all_columns.remove(target_variable)
    for non_numeric_att in non_numeric_att_list:
        all_columns.remove(non_numeric_att)

    # pandas dataset
    original_data = pd.read_csv(StringIO(data_file))

    # ignore very small datasets
    n_rows = original_data.shape[0]
    if n_rows < params['min_number_records']:
        print('The following dataset does not have the minimum number of records: %s' % data_name)
        return result

    # generating the key column for the data
    key_column = [
        ''.join(
            [random.choice(string.ascii_letters + string.digits) for n in range(10)]
        ) for _ in range(n_rows)
    ]

    # information for saving datasets
    identifier = str(uuid.uuid4())
    create_dir(os.path.join(output_dir, identifier), cluster_execution)

    # creating and saving query and candidate datasets
    query_data_paths = list()
    candidate_data_paths = list()
    for n in list(n_columns_query_dataset):
        # randomly choose the columns
        columns = list(np.random.choice(
            all_columns,
            n,
            replace=False
        ))

        # generate query data
        query_data_paths += generate_data_from_columns(
            original_data=original_data,
            columns=columns + [target_variable],
            column_metadata=column_metadata,
            key_column=key_column,
            params=params,
            dataset_path=os.path.join(output_dir, identifier),
            dataset_name=data_name,
            id_=len(query_data_paths),
            query=True
        )

        # generate candidate data
        candidate_data_paths += generate_data_from_columns(
            original_data=original_data,
            columns=list(set(all_columns).difference(set(columns))),
            column_metadata=column_metadata,
            key_column=key_column,
            params=params,
            dataset_path=os.path.join(output_dir, identifier),
            dataset_name=data_name,
            id_=len(candidate_data_paths),
            query=False
        )

    return [(
        identifier,
        original_data.columns[target_variable],
        query_data_paths,
        candidate_data_paths
    )]


 # # generating name and setting index
 #    query_data_names = list()
 #    for i in range(len(query_data)):
 #        name = identifier + os.path.sep + 'query_%s_%d.csv' % (data_name, i)
 #        query_data_names.append(name)
 #        query_data[i].set_index(
 #            'key-for-ranking',
 #            drop=True,
 #            inplace=True
 #        )
 #    candidate_data_names = list()
 #    for i in range(len(candidate_data)):
 #        name = identifier + os.path.sep + 'candidate_%s_%d.csv' % (data_name, i)
 #        candidate_data_names.append(name)
 #        candidate_data[i].set_index(
 #            'key-for-ranking',
 #            drop=True,
 #            inplace=True
 #        )

 #    target_variable_name = original_data.columns[target_variable]

 #    # doing joins and computing performance scores
 #    for i in range(len(query_data)):
 #        q_data = query_data[i]
 #        q_data_name = query_data_names[i]

 #        # build model on query data only
 #        score_before = get_performance_score(
 #            q_data,
 #            target_variable_name,
 #            algorithm
 #        )

 #        query_results = list()

 #        for j in range(len(candidate_data)):
 #            c_data = candidate_data[j]
 #            c_data_name = candidate_data_names[j]

 #            # join dataset
 #            join_ = q_data.join(
 #                c_data,
 #                how='left',
 #                rsuffix='_r'
 #            )
 #            join_.dropna(inplace=True)

 #            if join_.shape[0] < 50:
 #                continue

 #            # build model on joined data
 #            score_after = get_performance_score(
 #                join_,
 #                target_variable_name,
 #                algorithm
 #            )

 #            query_results.append((c_data_name, c_data, score_before, score_after))

 #        result.append(((q_data_name, q_data, target_variable_name), query_results))


def generate_query_and_candidate_datasets_negative_examples(query_dataset, candidate_set, max_number_random_candidates, params, identifier):
    """Generates query and cadidate datasets for negative examples.
    """

    # params
    output_dir = params['output_directory']
    training_data_file = params['training_data_file']

    # create identifier directory
    os.makedirs(os.path.join(output_dir, identifier))

    # query dataset
    query_dataset_name = query_dataset.split(",")[0]
    target_variable_name = query_dataset.split(",")[1]
    score_before = query_dataset.split(",")[2]
    query_dataset_source = "_".join(query_dataset_name.split("_")[1:-1])

    # candidate datasets
    candidate_datasets = list(candidate_set)
    random.shuffle(candidate_datasets)

    training_data = open(training_data_file, 'a')

    size = 0
    for i in range(len(candidate_datasets)):
        candidate_dataset = candidate_datasets[i]
        candidate_dataset_source = "_".join(candidate_dataset.split("_")[1:-1])
        if query_dataset_source == candidate_dataset_source:
            continue

        query_dataframe = pd.read_csv(os.path.join(output_dir, query_dataset_name))
        query_dataframe.drop(columns=['key-for-ranking'], inplace=True)
        candidate_dataframe = pd.read_csv(os.path.join(output_dir, candidate_dataset))
        candidate_dataframe.drop(columns=['key-for-ranking'], inplace=True)

        # generating the key column for the data and setting the index
        n_rows = max(query_dataframe.shape[0], candidate_dataframe.shape[0])
        key_column = [
            ''.join(
                [random.choice(string.ascii_letters + string.digits) for n in range(10)]
            ) for _ in range(n_rows)
        ]

        query_name = os.path.splitext(os.path.basename(query_dataset_name))[0] + "_" + str(size) + ".csv"
        query_dataframe.insert(
            0,
            'key-for-ranking',
            key_column[:min(len(key_column), query_dataframe.shape[0])]
        )
        query_dataframe.set_index(
            'key-for-ranking',
            drop=True,
            inplace=True
        )

        candidate_name = os.path.splitext(os.path.basename(candidate_dataset))[0] + "_" + str(size) + ".csv"
        candidate_dataframe.insert(
            0,
            'key-for-ranking',
            key_column[:min(len(key_column), candidate_dataframe.shape[0])]
        )
        candidate_dataframe.set_index(
            'key-for-ranking',
            drop=True,
            inplace=True
        )

        # join dataset
        join_ = query_dataframe.join(
            candidate_dataframe,
            how='left',
            rsuffix='_r'
        )
        join_.dropna(inplace=True)

        if join_.shape[0] < 50:
            continue

        # build model on joined data
        score_after = get_performance_score(
            join_,
            target_variable_name,
            params['regression_algorithm']
        )
        
        size += 1

        # saving data
        query_dataframe.to_csv(
            open(os.path.join(output_dir, identifier, query_name), 'w'),
            index=True
        )
        candidate_dataframe.to_csv(
            open(os.path.join(output_dir, identifier, candidate_name), 'w'),
            index=True
        )

        training_data.write('%s,%s,%s,%s,%.10f\n' % (os.path.join(identifier, query_name),
                                                     target_variable_name,
                                                     os.path.join(identifier, candidate_name),
                                                     score_before,
                                                     score_after))

        if size >= max_number_random_candidates:
            break

    training_data.close()

    return


def generate_data_from_columns(original_data, columns, column_metadata, key_column,
                               params, dataset_path, dataset_name, id_, query=True):
    """Generates datasets from the original data using only the columns specified
    in 'columns'. It also saves the datasets to disk.
    """

    paths = list()

    for col in column_metadata:
        if 'real' in column_metadata[col] or 'integer' in column_metadata[col]:
            original_data[original_data.columns[col]] = pd.to_numeric(original_data[original_data.columns[col]], errors='coerce')
    column_names = [original_data.columns[i] for i in columns]
    new_data = original_data[column_names].copy()
    new_data.fillna(value=0, inplace=True)
    new_data.insert(0, 'key-for-ranking', key_column)

    # saving dataset
    name_identifier = 'query' if query else 'candidate'
    name = '%s_%s_%d.csv' % (name_identifier, dataset_name, id_)
    save_file(
        os.path.join(dataset_path, name),
        new_data.to_csv(index=False),
        params['cluster'],
        params['hdfs_address'],
        params['hdfs_user']
    )
    paths.append(os.path.join(dataset_path, name))

    # number of times to randomly remove records
    n_times_remove_records = random.randint(1, params['max_times_records_removed'])
    for i in range(n_times_remove_records):

        # number of records to remove
        n_records_remove = random.randint(
            1,
            int(params['max_ratio_records_removed'] * original_data.shape[0])
        )

        # rows to remove
        drop_indices = np.random.choice(new_data.index, n_records_remove, replace=False)

        # ignoring small datasets
        if (new_data.shape[0] - len(drop_indices)) < params['min_number_records']:
            continue

        # saving dataset
        name = '%s_%s_%d.csv' % (name_identifier, dataset_name, id_ + i + 1)
        save_file(
            os.path.join(dataset_path, name),
            new_data.drop(drop_indices).to_csv(index=False),
            params['cluster'],
            params['hdfs_address'],
            params['hdfs_user']
        )
        paths.append(os.path.join(dataset_path, name))

    return paths


def get_performance_score(data, target_variable_name, algorithm):
    """Builds a model using data to predict the target variable,
    return the corresponding r2 score.
    """

    X_train, X_test, y_train, y_test = train_test_split(
        data.drop(target_variable_name, axis=1),
        data[target_variable_name],
        test_size=0.33,
        random_state=42
    )

    yfit = None

    if algorithm == 'random forest':
        forest = RandomForestRegressor(n_estimators=100, random_state=42)
        forest.fit(X_train, y_train)
        yfit = forest.predict(X_test)
    elif algorithm == 'linear':
        linear_r = LinearRegression(normalize=True)
        linear_r.fit(X_train, y_train)
        yfit = linear_r.predict(X_test)

    return r2_score(y_test, yfit)
    

if __name__ == '__main__':

    start_time = time.time()

    # Spark context
    conf = SparkConf().setAppName("Data Generation")
    sc = SparkContext(conf=conf)

    # parameters
    params = json.load(open(".params.json"))
    dir_ = params['datasets_directory']
    output_dir = params['output_directory']
    training_data_file = params['training_data_file']
    cluster_execution = params['cluster']
    hdfs_address = params['hdfs_address']
    hdfs_user = params['hdfs_user']

    create_dir(output_dir, cluster_execution)

    # dataset files
    dataset_files = list()
    if cluster_execution:
        # if executing on cluster, need to read from HDFS
        hdfs_client = InsecureClient(hdfs_address, user=hdfs_user)
        for dataset_path in hdfs_client.list(dir_):
            for f in hdfs_client.list(os.path.join(dir_, dataset_path)):
                dataset_files.append(os.path.join(dir_, dataset_path, f))
    else:
        # if executing locally, need to read from local file
        for dataset in os.listdir(dir_):
            if dataset == '.DS_Store':
                # ignoring .DS_Store on Mac
                continue
            data_path = os.path.join(
                dir_,
                dataset,
                dataset + '_dataset',
                'tables',
                'learningData.csv')
            dataset_doc = os.path.join(
                dir_,
                dataset,
                dataset + '_dataset',
                'datasetDoc.json')
            problem_doc = os.path.join(
                dir_,
                dataset,
                dataset + '_problem',
                'problemDoc.json')
            dataset_files.append(data_path)
            dataset_files.append(dataset_doc)
            dataset_files.append(problem_doc)

    all_files = sc.parallelize(dataset_files)

    # grouping files from same dataset
    # (dataset_name, [('learningData.csv', path),
    #                 ('datasetDoc.json', path),
    #                 ('problemDoc.json', path)])
    all_files = all_files.map(lambda x: organize_dataset_files(x, cluster_execution))
    all_files = all_files.reduceByKey(lambda x1, x2: x1 + x2)

    # generating query and candidate datasets for positive examples
    query_and_candidate_data_positive = all_files.flatMap(
        lambda x: generate_query_and_candidate_datasets_positive_examples(x, params)
    ).persist(StorageLevel.MEMORY_AND_DISK)

    # total number of positive examples
    n_positive_examples = query_and_candidate_data_positive.map(
        lambda x: len(x[2]) * len(x[3])
    ).reduce(
        lambda x, y: x + y
    )

    # generating query and candidate datasets for negative examples
    #   number of negative examples should be similar to the number of
    #   positive examples
    query_and_candidate_data = query_and_candidate_data_positive.collect()

    # # generating negative examples

    # query_set = set()
    # candidate_set = set()

    # training_data = open(training_data_file, 'r')
    # size_training_data = 0
    # line = training_data.readline()
    # while line != '':
    #     size_training_data += 1
    #     line_elems = line.split(",")
    #     query_set.add(",".join(line_elems[:2] + [line_elems[3]]))
    #     candidate_set.add(line_elems[2])
    #     line = training_data.readline()
    # training_data.close()

    # for query_dataset in query_set:

    #     generate_negative_training_data(
    #         query_dataset,
    #         candidate_set,
    #         size_training_data/len(query_set),
    #         params,
    #         str(id_)
    #     )

    #     id_ += 1

    # print("Duration: %.4f seconds" % (time.time() - start_time))
