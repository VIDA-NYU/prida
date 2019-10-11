#!/usr/bin/env python3
import json
import numpy as np
import os
import pandas as pd
from pyspark import SparkContext
import random
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import string
import sys
import time
import uuid

if sys.version_info[0] < 3: 
    from StringIO import StringIO
else:
    from io import StringIO


def generate_positive_training_data(input_dataset, regression_algorithm):
    """Generates training data, including query and cadidate datasets,
    and the corresponding performance scores, from a single dataset.

    The format of the input_dataset is as follows:

      (dataset_name, [('learningData.csv', ...),
                      ('datasetDoc.json', ...),
                      ('problemDoc.json', ...)])
    """

    result = list()

    # retrieving some dataset information

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
            data_file = d[1]
            continue
        if d[0] == 'datasetDoc.json':
            dataset_doc = json.loads(d[1])
            continue
        if d[0] == 'problemDoc.json':
            problem_doc = json.loads(d[1])

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

    # params
    algorithm = params['regression_algorithm']
    max_times_break_data_vertical = params['max_times_break_data_vertical']

    # non-numeric attributes
    n_non_numeric_att = 0
    non_numeric_att_list = list()
    for col in column_metadata:
        if 'real' not in column_metadata[col] and 'integer' not in column_metadata[col]:
            n_non_numeric_att += 1
            non_numeric_att_list.append(col)

    if target_variable in non_numeric_att_list:
        print('The following dataset has a non-numerical target variable: %s' % data_name)
        return result

    # removing target variable and non-numeric attributes
    n_columns_left = len(column_metadata) - 1 - n_non_numeric_att
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
    all_columns = list(range(0, len(column_metadata)))
    all_columns.remove(target_variable)
    for non_numeric_att in non_numeric_att_list:
        all_columns.remove(non_numeric_att)

    # pandas dataset
    original_data = pd.read_csv(StringIO(data_file))

    # generating the key column for the data
    n_rows = original_data.shape[0]
    key_column = [
        ''.join(
            [random.choice(string.ascii_letters + string.digits) for n in range(10)]
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
            original_data,
            columns + [target_variable],
            column_metadata,
            key_column,
            params
        )

        # generate candidate data
        candidate_data += generate_data_from_columns(
            original_data,
            list(set(all_columns).difference(set(columns))),
            column_metadata,
            key_column,
            params
        )

    identifier = str(uuid.uuid4())

    # generating name and setting index
    query_data_names = list()
    for i in range(len(query_data)):
        name = identifier + os.path.sep + 'query_%s_%d.csv' % (data_name, i)
        query_data_names.append(name)
        query_data[i].set_index(
            'key-for-ranking',
            drop=True,
            inplace=True
        )
    candidate_data_names = list()
    for i in range(len(candidate_data)):
        name = identifier + os.path.sep + 'candidate_%s_%d.csv' % (data_name, i)
        candidate_data_names.append(name)
        candidate_data[i].set_index(
            'key-for-ranking',
            drop=True,
            inplace=True
        )

    target_variable_name = original_data.columns[target_variable]

    # doing joins and computing performance scores
    for i in range(len(query_data)):
        q_data = query_data[i]
        q_data_name = query_data_names[i]

        # build model on query data only
        score_before = get_performance_score(
            q_data,
            target_variable_name,
            algorithm
        )

        query_results = list()

        for j in range(len(candidate_data)):
            c_data = candidate_data[j]
            c_data_name = candidate_data_names[j]

            # join dataset
            join_ = q_data.join(
                c_data,
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
                algorithm
            )

            query_results.append((c_data_name, c_data, score_before, score_after))

        result.append(((q_data_name, q_data, target_variable_name), query_results))

    return result


def generate_data_from_columns(original_data, columns, column_metadata, key_column, params):
    """Generates datasets from the original data using only the columns specified
    in 'columns'.
    """

    all_data = list()

    for col in column_metadata:
        if 'real' in column_metadata[col] or 'integer' in column_metadata[col]:
            original_data[original_data.columns[col]] = pd.to_numeric(original_data[original_data.columns[col]], errors='coerce')
    column_names = [original_data.columns[i] for i in columns]
    new_data = original_data[column_names].copy()
    new_data.fillna(value=0, inplace=True)
    # new_data.dropna(inplace=True)
    new_data.insert(0, 'key-for-ranking', key_column)

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


def get_candidate_files(record):
    """Extracts the files for candidate datasets.
    """
    names = list()
    for v in x[1]:
        names.append((v[0], v[1]))
    return names


def generate_negative_training_data(query_dataset, candidate_set, max_number_random_candidates, params, identifier):
    """Generates training data by randomly choosing candidate datasets
    for the input query dataset.
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

    if algorithm == 'linear':
        forest = RandomForestRegressor(n_estimators=100, random_state=42)
        forest.fit(X_train, y_train)
        yfit = forest.predict(X_test)
    elif algorithm == 'random forest':
        linear_r = LinearRegression(normalize=True)
        linear_r.fit(X_train, y_train)
        yfit = linear_r.predict(X_test)

    return r2_score(y_test, yfit)
    

if __name__ == '__main__':

    start_time = time.time()

    # Spark context
    sc = SparkContext()

    # parameters
    params = json.load(open(".params.json"))
    dir_ = params['datasets_directory']
    output_dir = params['output_directory']
    training_data_file = params['training_data_file']

    # dataset files
    data_files = list()
    hadoop = sc._jvm.org.apache.hadoop
    fs = hadoop.fs.FileSystem
    conf = hadoop.conf.Configuration()
    path = hadoop.fs.Path(dir_)
    for f in fs.get(conf).listStatus(path):
        data_files.append(f.getPath().toString())

    # loading all files
    data_rdd = list()
    for data_file in data_files:
        # dataset = os.path.basename(os.path.normpath(data_file))
        data_rdd.append(sc.wholeTextFiles(data_file))
    all_files = sc.union(data_rdd)

    # grouping files from same dataset
    # (dataset_name, [('learningData.csv', ...),
    #                 ('datasetDoc.json', ...),
    #                 ('problemDoc.json', ...)])
    all_files = all_files.map(lambda x: (x[0].split(os.path.sep)[-2], [(os.path.basename(os.path.normpath(x[0])), x[1])]))
    all_files = all_files.reduceByKey(lambda x1, x2: x1 + x2)

    # generating positive examples
    positive_examples = all_files.flatMap(lambda x: generate_positive_training_data(x, params))

    # saving files
    query_filenames = positive_examples.map(lambda x: x[0][0]).collect()
    candidate_files = positive_examples.flatMap(lambda x: get_candidate_files(x))
    candidate_filenames = candidate_files.map(lambda x: x[0]).collect()
    for query_filename in query_filenames:
        identifier_dir = query_filename.split(os.path.sep)[0]
        fs.get(conf).mkdirs(hadoop.fs.Path(output_dir + os.path.sep + identifier_dir))
        positive_examples.filter(lambda x: x[0][0] == query_filename).map(lambda x: x[0][1]).saveAsTextFile(output_dir + os.path.sep + query_filename)
    for candidate_filename in candidate_filenames:
        candidate_files.filter(lambda x: x[0] == candidate_filename).map(lambda x: x[1]).saveAsTextFile(output_dir + os.path.sep + candidate_filename)

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
