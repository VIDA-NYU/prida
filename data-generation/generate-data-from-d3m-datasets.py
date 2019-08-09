#!/usr/bin/env python3
import json
import numpy as np
import os
import pandas as pd
import random
import string


def retrieve_dataset_information(dataset_dir):
    """Retrieves some information about a D3M dataset,
    including data path, problem type, target variable,
    a boolean value that indicates whether the dataset
    is composed by multiple data, and data type for
    each column of the learningData file.
    """

    problem_type = None
    target_variable = None
    multiple_data = False
    column_metadata = dict()

    dataset_name = os.path.basename(os.path.normpath(dataset_dir))
    
    # data path
    data_path = os.path.join(
        dataset_dir,
        dataset_name + '_dataset',
        'tables',
        'learningData.csv')

    dataset_doc =  json.load(open(os.path.join(
        dataset_dir,
        dataset_name + '_dataset',
        'datasetDoc.json')))
    problem_doc = json.load(open(os.path.join(
        dataset_dir,
        dataset_name + '_problem',
        'problemDoc.json')))

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

    return dict(data_path=data_path,
                problem_type=problem_type,
                target_variable=target_variable,
                multiple_data=multiple_data,
                column_metadata=column_metadata)


def generate_training_data(data_name, data_path, target_variable, column_metadata, params):
    """Generates training data, including query and cadidate datasets,
    and the corresponding performance scores.
    """

    # params
    output_dir = params['output_directory']
    training_data_file = params['training_data_file']
    algorithm = params['regression_algorithm']

    n_columns_left = len(column_metadata) - 2  # removing target variable and 'd3mIndex'
    # if there is only one column left, there is no way to
    # generate both query and candidate datasets
    if n_columns_left <= 1:
        return

    # potential numbers of columns in a query dataset
    # for instance, if a dataset has 3 columns left (ignoring the target variable),
    #   a query dataset could have either 1 or 2 of these columns (so that the
    #   candidate dataset can have 1 or 2 columns, respectively)
    # in this example, n_columns_query_dataset = [1, 2]
    n_potential_columns_query_dataset = list(range(1, n_columns_left))

    # maximum number of times that the original data will be vertically broken into
    #   multiple datasets
    n_vertical_data = np.random.choice(
        list(range(1, min(params['max_times_break_data_vertical'], n_columns_left)))
    )

    # number of columns for each time the data is vertically broken
    n_columns_query_dataset = np.random.choice(
        n_potential_columns_query_dataset,
        n_vertical_data,
        replace=False
    )

    # list of column indices
    all_columns = list(range(1, len(column_metadata)))
    all_columns.remove(target_variable)

    # generating the key column for the data
    n_rows = pd.read_csv(data_path).shape[0]
    key_column = [
        ''.join(
            [random.choice(string.ascii_letters + string.digits) for n in range(5)]
        ) for _ in range(n_rows)
    ]

    query_data = list()
    candidate_data = list()

    for n in list(n_columns_query_dataset):
        # randomly choose the columns
        columns = list(np.random.choice(
            all_columns,
            n,
            replace=False
        ))

        # generate query data
        query_data += generate_data_from_columns(
            data_path,
            columns + [target_variable],
            key_column,
            params
        )

        # generate candidate data
        candidate_data += generate_data_from_columns(
            data_path,
            list(set(all_columns).difference(set(columns))),
            key_column,
            params
        )

    return


def generate_data_from_columns(data_path, columns, key_column, params):
    """Generates datasets from the original data using only the columns specified
    in 'columns'.
    """

    all_data = list()

    original_data = pd.read_csv(data_path)
    column_names = [original_data.columns[i] for i in columns]
    new_data = original_data[column_names]
    new_data.insert(0, 'key', key_column)

    all_data.append(new_data)

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
        all_data.append(new_data.drop(drop_indices))

    return all_data
    

if __name__ == '__main__':

    params = json.load(open('params.json'))

    dir_ = params['datasets_directory']
    for dataset in os.listdir(dir_):
        info = retrieve_dataset_information(os.path.join(dir_, dataset))
        # regression problems only
        if info['problem_type'] != 'regression':
            continue
        # single data tables only
        if info['multiple_data']:
            continue

        generate_training_data(
            dataset,
            info['data_path'],
            info['target_variable'],
            info['column_metadata'],
            params
        )
