from hdfs import InsecureClient
from io import StringIO
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

# number of additional candidates per query
NUMBER_ADDITIONAL_DATASETS = 175


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


def create_dir(file_path, hdfs_client=None, use_hdfs=False):
    """Creates a new directory specified by file_path.
    Returns True on success.
    """

    if use_hdfs:
        if hdfs_client.status(file_path, strict=False):
            print('[WARNING] Directory already exists: %s' % file_path)
            hdfs_client.delete(file_path, recursive=True, skip_trash=True)
        hdfs_client.makedirs(file_path)
    else:
        if os.path.exists(file_path):
            print('[WARNING] Directory already exists: %s' % file_path)
            shutil.rmtree(file_path)
        os.makedirs(file_path)
    return True


def save_file(file_path, content, hdfs_client=None, use_hdfs=False):
    """Opens a file for write and returns its corresponding file object.
    """

    if use_hdfs:
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


def delete_dir(file_path, hdfs_client=None, use_hdfs=False):
    """Deletes a directory.
    """

    if use_hdfs:
        if hdfs_client.status(file_path, strict=False):
            hdfs_client.delete(file_path, recursive=True, skip_trash=True)
    else:
        if os.path.exists(file_path):
            shutil.rmtree(file_path)
    # print('[INFO] File %s saved!' % file_path)


def generate_candidate_datasets_negative_examples(query_dataset, target_variable, candidate_datasets, params):
    """Generates candidate datasets for negative examples.
    This is necessary because query and candidate datasets must match for the join;
    therefore, we need to re-create the key column.
    """

    # print('[INFO] Creating negative examples with dataset %s ...' % query_dataset)

    # accumulator
    global new_combinations_counter

    new_candidate_datasets = list()

    # params
    output_dir = params['new_datasets_directory']
    files_dir = os.path.join(output_dir, 'files-test-data')
    cluster_execution = params['cluster']
    hdfs_address = params['hdfs_address']
    hdfs_user = params['hdfs_user']

    # HDFS Client
    hdfs_client = None
    if cluster_execution:
        hdfs_client = InsecureClient(hdfs_address, user=hdfs_user)

    # information for saving datasets
    identifier = str(uuid.uuid4())
    identifier_dir = os.path.join(files_dir, identifier)
    create_dir(identifier_dir, hdfs_client, cluster_execution)

    # reading query dataset
    query_data_str = read_file(query_dataset, hdfs_client, cluster_execution)
    query_data = pd.read_csv(StringIO(query_data_str))
    query_data_key_column = list(query_data['key-for-ranking'])

    new_combinations_counter += len(candidate_datasets)

    id_ = 0
    for i in range(len(candidate_datasets)):

        candidate_dataset = candidate_datasets[i]

        # reading candidate dataset
        candidate_data_str = read_file(candidate_dataset, hdfs_client, cluster_execution)
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
            hdfs_client,
            cluster_execution
        )

        new_candidate_datasets.append(file_path)

        id_ += 1

    # saving query dataset
    query_dataset_path = os.path.join(identifier_dir, os.path.basename(query_dataset))
    save_file(
        query_dataset_path,
        query_data.to_csv(index=False),
        hdfs_client,
        cluster_execution
    )

    # saving target information
    save_file(
        os.path.join(identifier_dir, '.target'),
        target_variable,
        hdfs_client,
        cluster_execution
    )

    # print('[INFO] Negative examples with dataset %s have been created and saved!' % query_dataset)
    return (query_dataset_path, target_variable, new_candidate_datasets)


def generate_performance_scores(query_dataset, target_variable, candidate_datasets, params):
    """Generates all the performance scores.
    """

    performance_scores = list()

    # params
    algorithm = params['regression_algorithm']
    cluster_execution = params['cluster']
    hdfs_address = params['hdfs_address']
    hdfs_user = params['hdfs_user']
    inner_join = params['inner_join']

    # HDFS Client
    hdfs_client = None
    if cluster_execution:
        # time.sleep(np.random.randint(1, 120))  # avoid opening multiple sockets at the same time
        hdfs_client = InsecureClient(hdfs_address, user=hdfs_user)

    # reading query dataset
    query_data_str = read_file(query_dataset, hdfs_client, cluster_execution)
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
        candidate_data_str = read_file(candidate_dataset, hdfs_client, cluster_execution)
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
        # strategies = ['mean', 'median', 'most_frequent']
        strategies = ['mean']  # using only mean for now
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
        # normalizing data first
        scaler_X = StandardScaler().fit(X_train)
        scaler_y = StandardScaler().fit(y_train.values.reshape(-1, 1))
        X_train = scaler_X.transform(X_train)
        y_train = scaler_y.transform(y_train.values.reshape(-1, 1))
        X_test = scaler_X.transform(X_test)
        y_test = scaler_y.transform(y_test.values.reshape(-1, 1))

        forest = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
            max_depth=len(data.columns)-1
        )
        forest.fit(X_train, y_train.ravel())
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
    conf = SparkConf().setAppName("Data Generation (Scores) for Test Data")
    sc = SparkContext(conf=conf)

    # counters
    existing_combinations_counter = 0
    new_combinations_counter = sc.accumulator(0)

    # parameters
    params = json.load(open(".params.json"))
    output_dir = params['new_datasets_directory']
    files_dir = os.path.join(output_dir, 'files-test-data')
    cluster_execution = params['cluster']
    hdfs_address = params['hdfs_address']
    hdfs_user = params['hdfs_user']

    # HDFS Client
    hdfs_client = None
    if cluster_execution:
        hdfs_client = InsecureClient(hdfs_address, user=hdfs_user)

    create_dir(files_dir, hdfs_client, cluster_execution)

    # reading test data
    #   assuming filename is 'test-data.csv'
    if not os.path.exists("test-data.csv"):
        print("Test data does not exist: test-data.csv")
        sys.exit(0)

    test_data = sc.parallelize(open("test-data.csv").readlines()).map(
        lambda x: x.split(',')
    ).map(
        lambda x: (x[0], x[1], x[2])
    ).persist(StorageLevel.MEMORY_AND_DISK)

    existing_combinations_counter = test_data.count()

    new_combinations = test_data.cartesian(test_data).filter(
        # filtering same query candidate
        lambda x: x[0][0] != x[1][0]
    ).map(
        # key => (query dataset, target variable)
        # val => [candidate dataset]
        lambda x: ((x[0][0], x[0][1]), [x[1][2]])
    ).reduceByKey(
        # concatenating lists of candidate datasets
        lambda x, y: x + y
    ).map(
        # (query dataset, target variable, random candidate datasets)
        lambda x: (x[0][0], x[0][1], list(
            np.random.choice(x[1], size=NUMBER_ADDITIONAL_DATASETS, replace=False)))
    ).persist(StorageLevel.MEMORY_AND_DISK)

    if not new_combinations.isEmpty():

        # getting performance scores
        performance_scores = new_combinations.map(
            lambda x: generate_candidate_datasets_negative_examples(x[0], x[1], x[2], params)
        ).flatMap(
            lambda x: generate_performance_scores(x[0], x[1], x[2], params)
        )

        # saving scores
        algorithm_name = params['regression_algorithm']
        if params['regression_algorithm'] == 'random forest':
            algorithm_name = 'random-forest'
        filename = os.path.join(output_dir, 'new-test-data-' + algorithm_name)
        delete_dir(filename, hdfs_client, cluster_execution)
        if not cluster_execution:
            filename = 'file://' + filename
        performance_scores.saveAsTextFile(filename)


    print('Duration: %.4f seconds' % (time.time() - start_time))
    print(' -- Configuration:')
    print('    . original_datasets_directory: %s' % params['original_datasets_directory'])
    print('    . new_datasets_directory: %s' % params['new_datasets_directory'])
    print('    . cluster: %s' % str(params['cluster']))
    print('    . hdfs_address: %s' % params['hdfs_address'])
    print('    . hdfs_user: %s' % params['hdfs_user'])
    print('    . ignore_first_attribute: %s' % str(params['ignore_first_attribute']))
    print('    . skip_dataset_creation: %s' % str(params['skip_dataset_creation']))
    print('    . skip_training_data: %s' % str(params['skip_training_data']))
    print('    . candidate_single_column: %s' % str(params['candidate_single_column']))
    print('    . regression_algorithm: %s' % params['regression_algorithm'])
    print('    . inner_join: %s' % str(params['inner_join']))
    print('    . min_number_records: %d' % params['min_number_records'])
    print('    . max_number_columns: %d' % params['max_number_columns'])
    print('    . max_times_break_data_vertical: %d' % params['max_times_break_data_vertical'])
    print('    . max_times_records_removed: %d' % params['max_times_records_removed'])
    print('    . max_percentage_noise: %d' % params['max_percentage_noise'])

    print(' -- N. existing combinations: %d' %existing_combinations_counter)
    print(' -- N. new combinations: %d' %new_combinations_counter.value)
