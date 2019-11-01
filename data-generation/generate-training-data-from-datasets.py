from hdfs import InsecureClient
from io import StringIO
from itertools import combinations
import json
import math
import numpy as np
import os
import pandas as pd
from pyspark import SparkConf, SparkContext, StorageLevel
import random
import re
import shutil
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, \
    mean_squared_log_error, median_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sys
import time
import uuid
from xgboost import XGBRegressor

# regex to take care of XGBoost ValueError
regex = re.compile(r"\[|\]|<", re.IGNORECASE)


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


def file_exists(file_path, use_hdfs=False, hdfs_address=None, hdfs_user=None):
    """Returns True if file exists.
    """

    if use_hdfs:
        hdfs_client = InsecureClient(hdfs_address, user=hdfs_user)
        if hdfs_client.status(file_path, strict=False):
            return True
    else:
        return os.path.exists(file_path)
    return False


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


def save_file(file_path, content, use_hdfs=False, hdfs_address=None, hdfs_user=None):
    """Opens a file for write and returns its corresponding file object.
    """

    if use_hdfs:
        hdfs_client = InsecureClient(hdfs_address, user=hdfs_user)
        if hdfs_client.status(file_path, strict=False):
            print('[WARNING] File already exists: %s' % file_path)
        with hdfs_client.write(file_path) as writer:
            writer.write(content.encode())
    else:
        if os.path.exists(file_path):
            print('[WARNING] File already exists: %s' % file_path)
        with open(file_path, 'w') as writer:
            writer.write(content)
    # print('[INFO] File %s saved!' % file_path)


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


def list_dir(file_path, use_hdfs=False, hdfs_address=None, hdfs_user=None):
    """Lists all the files inside the directory specified by file_path.
    """

    if use_hdfs:
        hdfs_client = InsecureClient(hdfs_address, user=hdfs_user)
        return hdfs_client.list(file_path)
    return os.listdir(file_path)


def generate_query_and_candidate_datasets_positive_examples(input_dataset, params):
    """Generates query and candidate datasets from a single dataset
    for positive examples.

    The format of input_dataset is as follows:

      (dataset_name, [('learningData.csv', path),
                      ('datasetDoc.json', path),
                      ('problemDoc.json', path)])
    """

    result = list()

    # accumulators
    global processed_datasets
    global no_appropriate_files
    global no_regression
    global multiple_files
    global no_numerical_target
    global no_enough_columns
    global no_enough_records
    global dataframe_exception

    # params
    max_times_break_data_vertical = params['max_times_break_data_vertical']
    ignore_first_attribute = params['ignore_first_attribute']
    candidate_single_column = params['candidate_single_column']
    output_dir = params['new_datasets_directory']
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
            dataset_doc = read_file(d[1], cluster_execution, hdfs_address, hdfs_user)
            continue
        if d[0] == 'problemDoc.json':
            problem_doc = read_file(d[1], cluster_execution, hdfs_address, hdfs_user)

    if not data_file or not dataset_doc or not problem_doc:
        print('[WARNING] The following dataset does not have the appropriate files: %s ' % data_name)
        no_appropriate_files += 1
        return result
    dataset_doc = json.loads(dataset_doc)
    problem_doc = json.loads(problem_doc)
    
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
        print('[WARNING] The following dataset does not belong to a regression problem: %s (%s)' % (data_name, problem_type))
        no_regression += 1
        return result
    # single data tables only
    if multiple_data:
        print('[WARNING] The following dataset is composed by multiple files: %s' % data_name)
        multiple_files += 1
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
        print('[WARNING] The following dataset has a non-numerical target variable: %s' % data_name)
        no_numerical_target += 1
        return result

    # removing target variable, non-numeric attributes, and first attribute (if it is to be ignored)
    n_columns_left = len(column_metadata) - 1 - n_non_numeric_att
    if ignore_first_attribute:
        n_columns_left -= 1
    # if there is only one column left, there is no way to
    # generate both query and candidate datasets
    if n_columns_left <= 1:
        print('[WARNING] The following dataset does not have enough columns for the data generation process: %s' % data_name)
        no_enough_columns += 1
        return result

    # potential numbers of columns in a query dataset
    # for instance, if a dataset has 3 columns left (ignoring the target variable),
    #   a query dataset could have either 1 or 2 of these columns (so that the
    #   candidate dataset can have 1 or 2 columns, respectively)
    # in this example, n_columns_query_dataset = [1, 2]
    n_potential_columns_query_dataset = list(range(1, n_columns_left))

    # maximum number of times that the original data will be vertically broken into
    #   multiple datasets
    n_vertical_data = min(max_times_break_data_vertical, n_columns_left)

    # number of columns for each time the data is vertically broken
    n_columns_query_dataset = np.random.choice(
        n_potential_columns_query_dataset,
        n_vertical_data,
        replace=True
    )

    # list of column indices
    range_start = 1 if ignore_first_attribute else 0
    all_columns = list(range(range_start, len(column_metadata)))
    all_columns.remove(target_variable)
    for non_numeric_att in non_numeric_att_list:
        all_columns.remove(non_numeric_att)

    # if candidate datasets should have a single column, try all combinations
    all_combinations = None
    if candidate_single_column:
        all_combinations = list(combinations(all_columns, n_columns_left - 1))
        n_columns_query_dataset = [n_columns_left - 1 for _ in all_combinations]

    # pandas dataset
    original_data = None
    try:
        original_data = pd.read_csv(StringIO(data_file))
    except Exception as e:
        print('[WARNING] The following dataset had an exception while parsing into a dataframe: %s (%s)' % (data_name, str(e)))
        dataframe_exception += 1
        return result

    # ignore very small datasets
    n_rows = original_data.shape[0]
    if n_rows < params['min_number_records']:
        print('[WARNING] The following dataset does not have the minimum number of records: %s' % data_name)
        no_enough_records += 1
        return result

    processed_datasets += 1

    # generating the key column for the data
    key_column = [str(uuid.uuid4()) for _ in range(n_rows)]

    # creating and saving query and candidate datasets
    # print('[INFO] Creating query and candidate data for dataset %s ...' % data_name)
    results = list()
    id_ = 0
    for n in list(n_columns_query_dataset):

        # information for saving datasets
        identifier = str(uuid.uuid4())
        identifier_dir = os.path.join(output_dir, 'files', identifier)
        create_dir(identifier_dir, cluster_execution, hdfs_address, hdfs_user)

        if not candidate_single_column:
            # randomly choose the columns
            columns = list(np.random.choice(
                all_columns,
                n,
                replace=False
            ))
        else:
            # get column indices from combinations
            columns = list(all_combinations[id_])

        # generate query data
        query_data_paths = generate_data_from_columns(
            original_data=original_data,
            columns=columns + [target_variable],
            column_metadata=column_metadata,
            key_column=key_column,
            params=params,
            dataset_path=identifier_dir,
            dataset_name=data_name,
            query=True
        )

        # generate candidate data
        candidate_data_paths = generate_data_from_columns(
            original_data=original_data,
            columns=list(set(all_columns).difference(set(columns))),
            column_metadata=column_metadata,
            key_column=key_column,
            params=params,
            dataset_path=identifier_dir,
            dataset_name=data_name,
            query=False
        )

        # saving target information
        save_file(
            os.path.join(identifier_dir, '.target'),
            original_data.columns[target_variable],
            params['cluster'],
            params['hdfs_address'],
            params['hdfs_user']
        )

        results.append((
            identifier,
            original_data.columns[target_variable],
            query_data_paths,
            candidate_data_paths,
            data_name
        ))

        id_ += 1

    # print('[INFO] Query and candidate data for dataset %s have been created and saved!' % data_name)
    return results


def generate_candidate_datasets_negative_examples(target_variable, query_dataset, candidate_datasets, params):
    """Generates candidate datasets for negative examples.
    This is necessary because query and candidate datasets must match for the join;
    therefore, we need to re-create the key column.
    """

    # print('[INFO] Creating negative examples with dataset %s ...' % query_dataset)

    new_candidate_datasets = list()

    # params
    output_dir = params['new_datasets_directory']
    cluster_execution = params['cluster']
    hdfs_address = params['hdfs_address']
    hdfs_user = params['hdfs_user']

    # information for saving datasets
    identifier = str(uuid.uuid4())
    identifier_dir = os.path.join(output_dir, 'files', identifier)
    create_dir(identifier_dir, cluster_execution, hdfs_address, hdfs_user)

    # reading query dataset
    query_data_str = read_file(query_dataset, cluster_execution, hdfs_address, hdfs_user)
    query_data = pd.read_csv(StringIO(query_data_str))
    query_data_key_column = list(query_data['key-for-ranking'])

    id_ = 0
    for candidate_dataset in candidate_datasets:

        # reading candidate dataset
        candidate_data_str = read_file(candidate_dataset, cluster_execution, hdfs_address, hdfs_user)
        candidate_data = pd.read_csv(StringIO(candidate_data_str))
        candidate_data.drop(columns=['key-for-ranking'], inplace=True)

        # generating extra key column entries, if necessary
        extra_key_column = list()
        if query_data.shape[0] < candidate_data.shape[0]:
            extra_key_column = [
                str(uuid.uuid4()) for _ in range(candidate_data.shape[0] - query_data.shape[0])
            ]

        # adding the key column to the candidate data
        min_size = min(query_data.shape[0], candidate_data.shape[0])
        candidate_data.insert(
            0,
            'key-for-ranking',
            query_data_key_column[:min_size] + extra_key_column
        )
        
        # saving candidate dataset
        dataset_name = "%s_%d.csv" % (os.path.splitext(os.path.basename(candidate_dataset))[0], id_)
        file_path = os.path.join(identifier_dir, dataset_name)
        save_file(
            file_path,
            candidate_data.to_csv(index=False),
            params['cluster'],
            params['hdfs_address'],
            params['hdfs_user']
        )

        new_candidate_datasets.append(file_path)

        id_ += 1

    # saving query dataset
    query_dataset_path = os.path.join(identifier_dir, os.path.basename(query_dataset))
    save_file(
        query_dataset_path,
        query_data.to_csv(index=False),
        params['cluster'],
        params['hdfs_address'],
        params['hdfs_user']
    )

    # saving target information
    save_file(
        os.path.join(identifier_dir, '.target'),
        target_variable,
        params['cluster'],
        params['hdfs_address'],
        params['hdfs_user']
    )

    # print('[INFO] Negative examples with dataset %s have been created and saved!' % query_dataset)
    return (target_variable, query_dataset_path, new_candidate_datasets)


def generate_data_from_columns(original_data, columns, column_metadata, key_column,
                               params, dataset_path, dataset_name, query=True):
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
    id_ = 0
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
    n_times_remove_records = params['max_times_records_removed']
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


def generate_performance_scores(query_dataset, target_variable, candidate_datasets, params):
    """Generates all the performance scores.
    The format of the output is the following:

      (query dataset, target variable name, candidate dataset,
       score before augmentation, score after augmentation)

    """

    performance_scores = list()

    # params
    algorithm = params['regression_algorithm']
    cluster_execution = params['cluster']
    hdfs_address = params['hdfs_address']
    hdfs_user = params['hdfs_user']
    inner_join = params['inner_join']

    # reading query dataset
    query_data_str = read_file(query_dataset, cluster_execution, hdfs_address, hdfs_user)
    query_data = pd.read_csv(StringIO(query_data_str))
    query_data.set_index(
        'key-for-ranking',
        drop=True,
        inplace=True
    )

    # build model on query data only
    _, scores_before = get_performance_scores(
        query_data,
        target_variable,
        algorithm,
        False
    )

    for candidate_dataset in candidate_datasets:

        # reading candidate dataset
        candidate_data_str = read_file(candidate_dataset, cluster_execution, hdfs_address, hdfs_user)
        candidate_data = pd.read_csv(StringIO(candidate_data_str))
        candidate_data.set_index(
            'key-for-ranking',
            drop=True,
            inplace=True
        )

        # join dataset
        join_ = query_data.join(
            candidate_data,
            how='left',
            rsuffix='_r'
        )
        if inner_join:
            join_.dropna(inplace=True)

        # build model on joined data
        # print('[INFO] Generating performance scores for query dataset %s and candidate dataset %s ...' % (query_dataset, candidate_dataset))
        imputation_strategy, scores_after = get_performance_scores(
            join_,
            target_variable,
            algorithm,
            not(inner_join)
        )
        # print('[INFO] Performance scores for query dataset %s and candidate dataset %s done!' % (query_dataset, candidate_dataset))

        performance_scores.append(
            generate_output_performance_data(
                query_dataset=query_dataset,
                target=target_variable,
                candidate_dataset=candidate_dataset,
                scores_before=scores_before,
                scores_after=scores_after,
                imputation_strategy=imputation_strategy
            )
        )

    return performance_scores


def get_performance_scores(data, target_variable_name, algorithm, missing_value_imputation):
    """Builds a model using data to predict the target variable,
    returning different performance metrics.
    """

    if missing_value_imputation:
        strategies = ['mean', 'median', 'most_frequent']
        scores = list()
        min_mean_absolute_error = math.inf
        min_strategy = ''
        for strategy in strategies:
            # imputation on data
            fill_NaN = SimpleImputer(missing_values=np.nan, strategy=strategy)
            new_data = pd.DataFrame(fill_NaN.fit_transform(data))
            new_data.columns = data.columns
            new_data.index = data.index

            # training and testing model
            strategy_scores = train_and_test_model(new_data, target_variable_name, algorithm)

            # always choosing the one with smallest mean absolute error
            if strategy_scores[0] < min_mean_absolute_error:
                min_mean_absolute_error = strategy_scores[0]
                min_strategy = strategy
                scores = [score for score in strategy_scores]

        return (min_strategy, scores)
    else:
        return (None, train_and_test_model(data, target_variable_name, algorithm))


def train_and_test_model(data, target_variable_name, algorithm):
    """Builds a model using data to predict the target variable.
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
    elif algorithm == 'sgd':
        # normalizing data first
        scaler_X = StandardScaler().fit(X_train)
        scaler_y = StandardScaler().fit(y_train.values.reshape(-1, 1))
        X_train = scaler_X.transform(X_train)
        y_train = scaler_y.transform(y_train.values.reshape(-1, 1))
        X_test = scaler_X.transform(X_test)
        y_test = scaler_y.transform(y_test.values.reshape(-1, 1))

        sgd = SGDRegressor()
        sgd.fit(X_train, y_train.ravel())
        yfit = sgd.predict(X_test)
    elif algorithm == 'xgboost':
        # taking care of 'ValueError: feature_names may not contain [, ] or <'
        X_train = replace_invalid_characters(X_train)
        X_test = replace_invalid_characters(X_test)

        xgboost_r = XGBRegressor(max_depth=5, objective='reg:squarederror', random_state=42)
        xgboost_r.fit(X_train, y_train)
        yfit = xgboost_r.predict(X_test)

    return [
        mean_absolute_error(y_test, yfit),
        mean_squared_error(y_test, yfit),
        median_absolute_error(y_test, yfit),
        r2_score(y_test, yfit),
    ]


def generate_output_performance_data(query_dataset, target, candidate_dataset,
                                     scores_before, scores_after, imputation_strategy=None):
    """Generates a training data record in JSON format.
    """

    return json.dumps(dict(
        query_dataset=os.path.sep.join(query_dataset.split(os.path.sep)[-2:]),
        target=target,
        candidate_dataset=os.path.sep.join(candidate_dataset.split(os.path.sep)[-2:]),
        imputation_strategy=imputation_strategy,
        mean_absolute_error=[scores_before[0], scores_after[0]],
        mean_squared_error=[scores_before[1], scores_after[1]],
        median_absolute_error=[scores_before[2], scores_after[2]],
        r2_score=[scores_before[3], scores_after[3]]
    ))


def replace_invalid_characters(data):
    """Takes care of the following error from XGBoost:
      ValueError: feature_names may not contain [, ] or <
    This function replaces these invalid characters with the string '_'

    From: https://stackoverflow.com/questions/48645846/pythons-xgoost-valueerrorfeature-names-may-not-contain-or/50633571
    """

    data.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in data.columns]
    return data
    

if __name__ == '__main__':

    start_time = time.time()

    # Spark context
    conf = SparkConf().setAppName("Data Generation")
    sc = SparkContext(conf=conf)

    # accumulators
    processed_datasets = sc.accumulator(0)
    no_appropriate_files = sc.accumulator(0)
    no_regression = sc.accumulator(0)
    multiple_files = sc.accumulator(0)
    no_numerical_target = sc.accumulator(0)
    no_enough_columns = sc.accumulator(0)
    no_enough_records = sc.accumulator(0)
    dataframe_exception = sc.accumulator(0)

    # parameters
    params = json.load(open(".params.json"))
    output_dir = params['new_datasets_directory']
    skip_dataset_creation = params['skip_dataset_creation']
    skip_training_data = params['skip_training_data']
    cluster_execution = params['cluster']
    hdfs_address = params['hdfs_address']
    hdfs_user = params['hdfs_user']

    # all query and candidate datasets
    #   format is the following:
    #   (target_variable, query_dataset_path, candidate_dataset_paths)
    query_candidate_datasets = sc.emptyRDD()

    n_positive_examples = 0
    n_negative_examples = 0
    if not skip_dataset_creation:

        dir_ = params['original_datasets_directory']
        create_dir(output_dir, cluster_execution, hdfs_address, hdfs_user)
        create_dir(os.path.join(output_dir, 'files'), cluster_execution, hdfs_address, hdfs_user)

        # dataset files
        dataset_files = list()
        if cluster_execution:
            for dataset_path in list_dir(dir_, cluster_execution, hdfs_address, hdfs_user):
                for f in list_dir(os.path.join(dir_, dataset_path), cluster_execution, hdfs_address, hdfs_user):
                    dataset_files.append(os.path.join(dir_, dataset_path, f))
        else:
            for dataset in list_dir(dir_, cluster_execution, hdfs_address, hdfs_user):
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
        if cluster_execution:
            all_files = all_files.repartition(100)

        # grouping files from same dataset
        # (dataset_name, [('learningData.csv', path),
        #                 ('datasetDoc.json', path),
        #                 ('problemDoc.json', path)])
        all_files = all_files.map(lambda x: organize_dataset_files(x, cluster_execution))
        all_files = all_files.reduceByKey(lambda x1, x2: x1 + x2)

        # generating query and candidate datasets for positive examples
        #   format is the following:
        #   (identifier, target_variable, query_dataset_paths, candidate_dataset_paths, dataset_name)
        query_and_candidate_data_positive = all_files.flatMap(
            lambda x: generate_query_and_candidate_datasets_positive_examples(x, params)
        ).persist(StorageLevel.MEMORY_AND_DISK)

        if not query_and_candidate_data_positive.isEmpty():

            # total number of query datasets
            n_query_datasets = query_and_candidate_data_positive.map(
                lambda x: len(x[2])
            ).reduce(
                lambda x, y: x + y
            )

            # total number of positive examples
            n_positive_examples = query_and_candidate_data_positive.map(
                lambda x: len(x[2]) * len(x[3])
            ).reduce(
                lambda x, y: x + y
            )

            # generating query and candidate dataset pairs for negative examples
            #   number of negative examples should be similar to the number of
            #   positive examples
            n_random_candidates_per_query = int(n_positive_examples / n_query_datasets)
            query_and_candidate_data_negative_tmp = query_and_candidate_data_positive.cartesian(
                query_and_candidate_data_positive
            ).filter(
                # filtering same identifier / dataset
                lambda x: x[0][0] != x[1][0] and x[0][4] != x[1][4]
            ).flatMap(
                # key => (target variable, query dataset)
                # val => list of candidate datasets
                lambda x: [((x[0][1], query_data), x[1][3]) for query_data in x[0][2]]
            ).reduceByKey(
                # concatenating lists of candidate datasets
                lambda x, y: x + y
            ).map(
                # (target variable, query dataset, random sample of other candidate datasets)
                lambda x: (x[0][0], x[0][1], list(np.random.choice(
                    x[1], size=min(n_random_candidates_per_query, len(x[1])), replace=False
                )))
            ).persist(StorageLevel.MEMORY_AND_DISK)

            # total number of negative examples
            n_negative_examples = query_and_candidate_data_negative_tmp.map(
                lambda x: len(x[2])
            ).reduce(
                lambda x, y: x + y
            )

            # generating candidate datasets for negative examples
            #   format is the following:
            #   (target_variable, query_dataset_path, candidate_dataset_paths)
            query_and_candidate_data_negative = query_and_candidate_data_negative_tmp.map(
                lambda x: generate_candidate_datasets_negative_examples(x[0], x[1], x[2], params)
            ).persist(StorageLevel.MEMORY_AND_DISK)

            query_candidate_datasets = sc.union([
                query_and_candidate_data_positive.flatMap(
                    lambda x: [(x[1], query, x[3]) for query in x[2]]
                ),
                query_and_candidate_data_negative
            ]).persist(StorageLevel.MEMORY_AND_DISK)

            save_file(
                os.path.join(output_dir, '.files-training-data'),
                '\n'.join(query_candidate_datasets.flatMap(
                    lambda x: [[x[0], x[1]] + x[2]]
                ).map(
                    lambda x: ','.join(x)
                ).collect()),
                params['cluster'],
                params['hdfs_address'],
                params['hdfs_user']
            )

    else:

        # datasets previously generated
        if file_exists(os.path.join(output_dir, '.files-training-data'), cluster_execution, hdfs_address, hdfs_user):
            query_candidate_datasets = sc.parallelize(
                read_file(os.path.join(output_dir, '.files-training-data'), cluster_execution, hdfs_address, hdfs_user).split('\n')
            ).map(
                lambda x: x.strip().split(',')
            ).map(
                lambda x: (x[0], x[1], x[2:])
            ).persist(StorageLevel.MEMORY_AND_DISK)
        else:
            data_by_identifier = list()
            for identifier in list_dir(os.path.join(output_dir, 'files'), cluster_execution, hdfs_address, hdfs_user):
                target_variable = None
                query_datasets = list()
                candidate_datasets = list()
                for f in list_dir(os.path.join(output_dir, 'files', identifier), cluster_execution, hdfs_address, hdfs_user):
                    file_path = os.path.join(output_dir, 'files', identifier, f)
                    if f.startswith('query_'):
                        query_datasets.append(file_path)
                    elif f.startswith('candidate_'):
                        candidate_datasets.append(file_path)
                    elif f == '.target':
                        target_variable = read_file(file_path, cluster_execution, hdfs_address, hdfs_user)
                data_by_identifier.append((target_variable, query_datasets, candidate_datasets))

            query_candidate_datasets = sc.parallelize(data_by_identifier).flatMap(
                lambda x: [(x[0], query, x[2]) for query in x[1]]
            ).persist(StorageLevel.MEMORY_AND_DISK)

        if cluster_execution:
            query_candidate_datasets = query_candidate_datasets.repartition(300)


    if not skip_training_data:

        if not skip_training_data and skip_dataset_creation:
            start_time = time.time()

        if not query_candidate_datasets.isEmpty():

            # getting performance scores
            performance_scores = query_candidate_datasets.flatMap(
                lambda x: generate_performance_scores(x[1], x[0], x[2], params)
            )

            # saving scores
            algorithm_name = params['regression_algorithm']
            if params['regression_algorithm'] == 'random forest':
                algorithm_name = 'random-forest'
            save_file(
                os.path.join(output_dir, 'training-data-' + algorithm_name),
                '\n'.join(performance_scores.collect()),
                params['cluster'],
                params['hdfs_address'],
                params['hdfs_user']
            )

    print('Duration: %.4f seconds' % (time.time() - start_time))
    if not skip_dataset_creation:
        print(' -- N. positive examples: %d' %n_positive_examples)
        print(' -- N. negative examples: %d' %n_negative_examples)
        print(' -- Processed datasets: %d' %processed_datasets.value)
        print(' -- Datasets w/o appropriate files: %d' %no_appropriate_files.value)
        print(' -- Datasets w/ no regression problem: %d' %no_regression.value)
        print(' -- Datasets w/ multiple data files: %d' %multiple_files.value)
        print(' -- Datasets w/o numeric targets: %d' %no_numerical_target.value)
        print(' -- Datasets w/o enough columns: %d' %no_enough_columns.value)
        print(' -- Datasets w/o enough records: %d' %no_enough_records.value)
        print(' -- Datasets w/ pandas.Dataframe exception: %d' %dataframe_exception.value)
