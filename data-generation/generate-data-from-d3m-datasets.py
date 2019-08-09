#!/usr/bin/env python3
import json
import os
import pandas as pd


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

        print("Dataset Name: %s" % dataset)
        print("Data Path: %s" % info['data_path'])
        print("Problem Type: %s" % info['problem_type'])
        print("Target Variable: %d" % info['target_variable'])
        print("Multiple Data? %s" % str(info['multiple_data']))
        print("Column Metadata: %r" % info['column_metadata'])

        pass
