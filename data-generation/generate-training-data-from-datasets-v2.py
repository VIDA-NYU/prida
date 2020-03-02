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
from scipy.special import comb
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


def get_dataset_name(file_path, cluster_execution):
    """Map function to get dataset name.
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

    return (dataset_name, file_name)


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


def create_dir(file_path, hdfs_client=None, use_hdfs=False):
    """Creates a new directory specified by file_path.
    Returns True on success.
    """

    if use_hdfs:
        if hdfs_client.status(file_path, strict=False):
            hdfs_client.delete(file_path, recursive=True, skip_trash=True)
        hdfs_client.makedirs(file_path)
    else:
        if os.path.exists(file_path):
            shutil.rmtree(file_path)
        os.makedirs(file_path)
    return True


def generate_query_and_candidate_splits(input_dataset, params):
    """Generates information about how to split the input dataset into
    query and candidate datasets.

    The format of input_dataset is as follows:

      (dataset_name, [('learningData.csv', data),
                      ('datasetDoc.json', data),
                      ('problemDoc.json', data)])
    """

    splits = list()

    # accumulators
    global processed_datasets
    global no_appropriate_files
    global no_regression
    global multiple_files
    global no_numerical_target
    global no_enough_columns
    global no_enough_records
    global many_records
    global dataframe_exception

    # params
    max_times_break_data_vertical = params['max_times_break_data_vertical']
    max_number_columns = params['max_number_columns']
    ignore_first_attribute = params['ignore_first_attribute']
    candidate_single_column = params['candidate_single_column']
    output_dir = params['new_datasets_directory']

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
            dataset_doc = d[1]
            continue
        if d[0] == 'problemDoc.json':
            problem_doc = d[1]

    if not data_file or not dataset_doc or not problem_doc:
        print('[WARNING] The following dataset does not have the appropriate files: %s ' % data_name)
        no_appropriate_files += 1
        return splits
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
        return splits
    # single data tables only
    if multiple_data:
        print('[WARNING] The following dataset is composed by multiple files: %s' % data_name)
        multiple_files += 1
        return splits

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
        return splits

    # removing target variable, non-numeric attributes, and first attribute (if it is to be ignored)
    n_columns_left = len(column_metadata) - 1 - n_non_numeric_att
    if ignore_first_attribute:
        n_columns_left -= 1
    # number of columns left will also depend on the maximum number of colums chosen by the user
    n_columns_left = min(n_columns_left, max_number_columns - 1)
    # if there is only one column left, there is no way to
    # generate both query and candidate datasets
    if n_columns_left <= 1:
        print('[WARNING] The following dataset does not have enough columns for the data generation process: %s' % data_name)
        no_enough_columns += 1
        return splits

    # potential numbers of columns in a query dataset
    # for instance, if a dataset has 3 columns left (ignoring the target variable),
    #   a query dataset could have either 1 or 2 of these columns (so that the
    #   candidate dataset can have 1 or 2 columns, respectively)
    # in this example, n_columns_query_dataset = [1, 2]
    n_potential_columns_query_dataset = list(range(1, n_columns_left))

    # maximum number of column combinations for the query dataset
    # for the example above, if the query dataset has 1 column,
    #   then it has 3 options as column (since there are 3 columns left),
    #   and if the query dataset has 2 columns,
    #   then it has 3 additional options (number of combinations of 3 columns taken 2 at a time),
    #   totalling 6
    n_column_combinations_query_dataset = 0
    size_column_combinations_query_dataset = list()
    if n_columns_left > max_times_break_data_vertical:
        # if the number of columns left is greater or equal than the desired number of vertical
        #  splits, no need to compute expensive combinations
        n_column_combinations_query_dataset = max_times_break_data_vertical
        size_column_combinations_query_dataset = list(range(1, n_columns_left))
    else:
        for k in n_potential_columns_query_dataset:
            n_comb = int(comb(n_columns_left, k))
            n_column_combinations_query_dataset += n_comb
            size_column_combinations_query_dataset += [k for _ in range(n_comb)]

    # maximum number of times that the original data will be vertically broken into
    #   multiple datasets
    n_vertical_data = min(max_times_break_data_vertical, n_column_combinations_query_dataset)

    # number of columns for each time the data is vertically broken
    n_columns_query_dataset = np.random.choice(
        size_column_combinations_query_dataset,
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
    original_data = None
    try:
        original_data = pd.read_csv(StringIO(data_file))
        original_data.name = data_name
    except Exception as e:
        print('[WARNING] The following dataset had an exception while parsing into a dataframe: %s (%s)' % (data_name, str(e)))
        dataframe_exception += 1
        return splits

    # ignore very small datasets
    n_rows = original_data.shape[0]
    if n_rows < params['min_number_records']:
        print('[WARNING] The following dataset does not have the minimum number of records: %s' % data_name)
        no_enough_records += 1
        return splits
    if n_rows > 900000:
        print('[WARNING] The following dataset has more than 900,000 records: %s' % data_name)
        many_records += 1
        return splits

    processed_datasets += 1

    # query dataset column combinations that have been processed
    seen_combinations = set()

    # creating splits for query and candidate datasets
    # print('[INFO] Creating split for query and candidate data for dataset %s ...' % data_name)
    for n in list(n_columns_query_dataset):

        # selecting columns for query dataset
        query_columns = list()
        seen = True
        while seen:
            # randomly choose the columns
            query_columns = list(np.random.choice(
                all_columns,
                n,
                replace=False
            ))
            if tuple(query_columns) not in seen_combinations:
                seen = False
        seen_combinations.add(tuple(query_columns))

        # query dataset columns
        query_columns = query_columns + [target_variable]

        # candidate dataset columns
        n_possible_columns = n_columns_left - n
        if candidate_single_column:
            n_possible_columns = 1
        candidate_columns = list(np.random.choice(
            list(set(all_columns).difference(set(query_columns))),
            n_possible_columns,
            replace=False
        ))

        splits.append((query_columns, candidate_columns))

    # print('[INFO] Query and candidate splits for dataset %s have been created!' % data_name)

    # target variable
    target_variable_name = original_data.columns[target_variable]

    return [((data_name, target_variable_name), data_file, column_metadata, splits)]


def generate_query_and_candidate_datasets_positive_examples(data, column_metadata, query_columns, candidate_columns, params):
    """Generates query and candidate datasets from the suggested split.
    """

    results = list()

    dataset = pd.read_csv(StringIO(data))
    n_rows = dataset.shape[0]

    # generating the key column for the data
    key_column = [str(uuid.uuid4()) for _ in range(n_rows)]

    # fixing data types
    for col in column_metadata:
        if 'real' in column_metadata[col] or 'integer' in column_metadata[col]:
            dataset[dataset.columns[col]] = pd.to_numeric(dataset[dataset.columns[col]], errors='coerce')
    
    # generate query data
    query_dataset = generate_data_from_columns(
        original_data=dataset,
        columns=query_columns,
        key_column=key_column,
        params=params,
        query=True
    )[0]

    # query data id
    query_data_id = str(uuid.uuid4())

    # generate candidate data
    candidate_datasets = generate_data_from_columns(
        original_data=dataset,
        columns=candidate_columns,
        key_column=key_column,
        params=params,
        query=False
    )

    # candidate data id
    candidate_data_id = list()
    for i in range(len(candidate_datasets)):
        candidate_data_id.append(str(uuid.uuid4()))

    return (query_data_id, query_dataset, candidate_data_id, candidate_datasets, len(query_columns), len(candidate_columns))


def generate_candidate_datasets_negative_examples(query_dataset, candidate_datasets, params):
    """Generates candidate datasets for negative examples.
    This is necessary because query and candidate datasets must match for the join;
    therefore, we need to re-create the key column.
    """

    new_candidate_datasets = list()

    query_data = pd.read_csv(StringIO(query_dataset))

    # query data id
    query_data_id = str(uuid.uuid4())

    # key column
    query_data_key_column = list(query_data['key-for-ranking'])

    # ratios of removed records
    max_ratio_records_removed = 1.0
    ratio_remove_record = list(np.arange(
        0, max_ratio_records_removed, max_ratio_records_removed/(len(candidate_datasets) + 1))
    )[1:]

    # candidate data id
    candidate_data_id = list()

    for i in range(len(candidate_datasets)):

        candidate_data = pd.read_csv(StringIO(candidate_datasets[i]))
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

        # # randomly removing records from candidate dataset
        # n_records_remove = int(ratio_remove_record[i] * min_size)
        # drop_indices = np.random.choice(candidate_data.index, n_records_remove, replace=False)
        # if (candidate_data.shape[0] - len(drop_indices)) >= params['min_number_records']:
        #     candidate_data = candidate_data.drop(drop_indices)

        new_candidate_datasets.append(candidate_data.to_csv(index=False))
        candidate_data_id.append(str(uuid.uuid4()))

    return (query_data_id, query_data.to_csv(index=False), candidate_data_id, new_candidate_datasets)


def generate_data_from_columns(original_data, columns, key_column, params, query=True):
    """Generates datasets from the original data using only the columns specified in 'columns'.
    """

    datasets = list()
    
    column_names = [original_data.columns[i] for i in columns]
    new_data = original_data[column_names].copy()
    new_data.fillna(value=0, inplace=True)
    new_data.insert(0, 'key-for-ranking', key_column)

    # saving dataset
    datasets.append(new_data.to_csv(index=False))

    # # only remove records from candidate datasets
    # if not query:

    #     # ratios of removed records
    #     max_ratio_records_removed = 1.0
    #     ratio_remove_record = list(np.arange(
    #         0, max_ratio_records_removed, max_ratio_records_removed/(params['max_times_records_removed'] + 1))
    #     )[1:]

    #     for i in range(len(ratio_remove_record)):

    #         # number of records to remove
    #         n_records_remove = int(ratio_remove_record[i] * original_data.shape[0])

    #         # rows to remove
    #         drop_indices = np.random.choice(new_data.index, n_records_remove, replace=False)

    #         # ignoring small datasets
    #         if (new_data.shape[0] - len(drop_indices)) < params['min_number_records']:
    #             continue

    #         # saving dataset
    #         datasets.append(new_data.drop(drop_indices).to_csv(index=False))

    return datasets


def generate_performance_scores(query_dataset_id, query_dataset, target_variable,
                                candidate_datasets_ids, candidate_datasets, params):
    """Generates all the performance scores.
    """

    performance_scores = list()

    # params
    algorithm = params['regression_algorithm']
    inner_join = params['inner_join']

    # reading query dataset
    query_data = pd.read_csv(StringIO(query_dataset))
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

    for i in range(len(candidate_datasets)):

        candidate_dataset = candidate_datasets[i]

        # reading candidate dataset
        candidate_data = pd.read_csv(StringIO(candidate_dataset))
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
                query_dataset=query_dataset_id,
                target=target_variable,
                candidate_dataset=candidate_datasets_ids[i],
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
            # max_depth=len(data.columns)-1
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
        query_dataset=query_dataset,
        target=target,
        candidate_dataset=candidate_dataset,
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
    many_records = sc.accumulator(0)
    dataframe_exception = sc.accumulator(0)

    # parameters
    params = json.load(open(".params.json"))
    output_dir = params['new_datasets_directory']
    skip_dataset_creation = params['skip_dataset_creation']
    skip_training_data = params['skip_training_data']
    cluster_execution = params['cluster']
    hdfs_address = params['hdfs_address']
    hdfs_user = params['hdfs_user']

    # HDFS Client
    hdfs_client = None
    if cluster_execution:
        hdfs_client = InsecureClient(hdfs_address, user=hdfs_user)

    # all query and candidate datasets
    #   format is the following:
    #   (query dataset id, target variable name, candidate dataset ids)
    query_candidate_datasets = sc.emptyRDD()

    # id to dataset mapping
    #   format is the following:
    #   (dataset id, dataset)
    dataset_id_to_data = sc.emptyRDD()

    n_training_datasets = 0
    n_test_datasets = 0
    n_positive_examples = 0
    n_negative_examples = 0
    if not skip_dataset_creation:

        dir_ = params['original_datasets_directory']
        create_dir(output_dir, hdfs_client, cluster_execution)

        # dataset files
        if cluster_execution:
            filename = os.path.join(dir_, '*', 'learningData.csv')
        else:
            filename = 'file://' + os.path.join(dir_, '*', '*_dataset', 'tables', 'learningData.csv')
        dataset_files = sc.wholeTextFiles(filename).map(
            lambda x: (get_dataset_name(x[0], cluster_execution), x[1])
        ).map(
            lambda x: (x[0][0], [(x[0][1], x[1])])
        )

        # datasetDoc files
        if cluster_execution:
            filename = os.path.join(dir_, '*', 'datasetDoc.json')
        else:
            filename = 'file://' + os.path.join(dir_, '*', '*_dataset', 'datasetDoc.json')
        dataset_doc_files = sc.wholeTextFiles(filename).map(
            lambda x: (get_dataset_name(x[0], cluster_execution), x[1])
        ).map(
            lambda x: (x[0][0], [(x[0][1], x[1])])
        )

        # problemDoc files
        if cluster_execution:
            filename = os.path.join(dir_, '*', 'problemDoc.json')
        else:
            filename = 'file://' + os.path.join(dir_, '*', '*_problem', 'problemDoc.json')
        problem_doc_files = sc.wholeTextFiles(filename).map(
            lambda x: (get_dataset_name(x[0], cluster_execution), x[1])
        ).map(
            lambda x: (x[0][0], [(x[0][1], x[1])])
        )

        # grouping files from same dataset
        # (dataset_name, [('learningData.csv', data),
        #                 ('datasetDoc.json', data),
        #                 ('problemDoc.json', data)])
        all_files = sc.union([dataset_files, dataset_doc_files, problem_doc_files])
        all_files = all_files.reduceByKey(lambda x1, x2: x1 + x2)

        if cluster_execution:
            all_files = all_files.repartition(372)

        # generating query and candidate splits for each dataset
        #   format is the following:
        #   ((datataset name, target variable name),
        #    dataset, column metadata, (query columns, candidate columns))
        query_and_candidate_splits = all_files.flatMap(
            lambda x: generate_query_and_candidate_splits(x, params)
        ).flatMap(
            lambda x: [(x[0], x[1], x[2], split) for split in x[3]]
        ).persist(StorageLevel.MEMORY_AND_DISK)

        if not query_and_candidate_splits.isEmpty():

            # creating query and candidate datasets
            #   format is the following:
            #   (dataset name, target variable name, query dataset id, query dataset,
            #    candidate datasets ids, candidate datasets, n. columns in query, n. columns in candidate)
            query_and_candidate_data = query_and_candidate_splits.map(
                lambda x: (x[0], generate_query_and_candidate_datasets_positive_examples(
                    x[1], x[2], x[3][0], x[3][1], params
                ))
            ).map(
                lambda x: (x[0][0], x[0][1], x[1][0], x[1][1], x[1][2], x[1][3], x[1][4], x[1][5])
            ).persist(StorageLevel.MEMORY_AND_DISK)

            # combinations of query and candidate datasets for positive examples
            #   format is the following:
            #   (dataset name, target variable name, query dataset id, candidate datasets ids,
            #    n. columns in query, n. columns in candidate)
            query_and_candidate_data_positive = query_and_candidate_data.map(
                lambda x: (x[0], x[1], x[2], x[4], x[6], x[7])
            ).persist(StorageLevel.MEMORY_AND_DISK)

            # id to dataset mapping
            dataset_id_to_data = query_and_candidate_data.flatMap(
                lambda x: [(x[2], x[3])] + [(x[4][i], x[5][i]) for i in range(len(x[4]))]
            ).persist(StorageLevel.MEMORY_AND_DISK)

            query_and_candidate_data.unpersist()

            # processed datasets
            all_datasets = list(set(query_and_candidate_data_positive.map(
                lambda x: x[0]
            ).collect()))

            # choosing datasets for training and testing
            training_datasets = list(np.random.choice(
                all_datasets,
                math.floor(0.7*len(all_datasets)),  ## 70% for training
                replace=False
            ))
            testing_datasets = list(set(all_datasets).difference(set(training_datasets)))
            n_training_datasets = len(training_datasets)
            n_test_datasets = len(testing_datasets)

            # filtering positive examples based on training and testing datasets
            query_and_candidate_data_positive_dict = dict()
            query_and_candidate_data_positive_dict['training'] = query_and_candidate_data_positive.filter(
                lambda x: x[0] in training_datasets
            ).persist(StorageLevel.MEMORY_AND_DISK)
            query_and_candidate_data_positive_dict['testing'] = query_and_candidate_data_positive.filter(
                lambda x: x[0] in testing_datasets
            ).persist(StorageLevel.MEMORY_AND_DISK)

            query_and_candidate_splits.unpersist()
            query_and_candidate_data_positive.unpersist()

            for key in query_and_candidate_data_positive_dict:

                query_and_candidate_data_positive_ = query_and_candidate_data_positive_dict[key]

                # total number of query datasets
                n_query_datasets = query_and_candidate_data_positive_.count()

                # total number of positive examples
                n_positive_examples = query_and_candidate_data_positive_.map(
                    lambda x: len(x[3])
                ).reduce(
                    lambda x, y: x + y
                )

                # generating query and candidate dataset pairs for negative examples
                #   number of negative examples should be similar to the number of
                #   positive examples
                n_random_candidates_per_query = int(n_positive_examples / n_query_datasets)

                # we do the cartesian product between query_and_candidate_data_positive_ and
                #   itself to choose a random set of candidates
                #   per query dataset (that does not belong to the same dataset)
                query_and_candidate_data_negative_tmp = query_and_candidate_data_positive_.cartesian(
                    query_and_candidate_data_positive_
                ).filter(
                    # filtering out same dataset and by max number of columns
                    lambda x: (
                        (x[0][0] != x[1][0]) and
                        (x[0][4] + x[1][5] <= params['max_number_columns'])
                    )
                ).map(
                    # key => (query dataset id, target variable)
                    # val => list of candidate ids
                    lambda x: ((x[0][2], x[0][1]), x[1][3])
                ).reduceByKey(
                    # concatenating lists of candidate datasets
                    lambda x, y: x + y
                ).map(
                    # ((query dataset id, target variable), random sample of other candidate datasets)
                    lambda x: (x[0], [x[1][i] for i in list(np.random.choice(
                        range(len(x[1])), size=min(n_random_candidates_per_query, len(x[1])), replace=False
                    ))])
                ).persist(StorageLevel.MEMORY_AND_DISK)

                # total number of negative examples
                n_negative_examples = query_and_candidate_data_negative_tmp.map(
                    lambda x: len(x[1])
                ).reduce(
                    lambda x, y: x + y
                )

                # generating candidate datasets for negative examples
                #   format is the following:
                #   (query dataset id, target variable, candidate dataset ids)
                query_and_candidate_data_negative = query_and_candidate_data_negative_tmp.map(
                    # first, let's use query dataset id as key
                    lambda x: (x[0][0], (x[0][1], x[1]))
                ).join(
                    # we get the query datasets
                    dataset_id_to_data
                ).map(
                    # (query dataset id, query dataset, target variable name, candidate dataset ids)
                    lambda x: (x[0], x[1][1], x[1][0][0], x[1][0][1])
                ).flatMap(
                    # first, let's use each candidate dataset id as key
                    lambda x: [(x[3][i], (x[0], x[1], x[2])) for i in range(len(x[3]))]
                ).join(
                    # we get the candidate datasets
                    dataset_id_to_data
                ).map(
                    # ((query dataset id, query dataset, target variable name), [candidate dataset])
                    lambda x: ((x[1][0][0], x[1][0][1], x[1][0][2]), [x[1][1]])
                ).reduceByKey(
                    # concatenating lists of candidate datasets
                    lambda x, y: x + y
                ).map(
                    # generating negative data
                    lambda x: (x[0][2], generate_candidate_datasets_negative_examples(x[0][1], x[1], params))
                ).map(
                    # (query dataset id, query dataset, target variable name, candidate datasets ids, candidate datasets)
                    lambda x: (x[1][0], x[1][1], x[0], x[1][2], x[1][3])
                ).persist(StorageLevel.MEMORY_AND_DISK)

                # updating id to dataset mapping
                dataset_id_to_data = sc.union([
                    dataset_id_to_data,
                    query_and_candidate_data_negative.flatMap(
                        lambda x: [(x[0], x[1])] + [(x[3][i], x[4][i]) for i in range(len(x[3]))]
                    )
                ]).persist(StorageLevel.MEMORY_AND_DISK)

                # (query dataset id, target variable name, candidate dataset ids)
                query_candidate_datasets_tmp = sc.union([
                    query_and_candidate_data_positive_.map(
                        lambda x: (x[2], x[1], x[3])
                    ),
                    query_and_candidate_data_negative.map(
                        lambda x: (x[0], x[2], x[3])
                    )
                ]).persist(StorageLevel.MEMORY_AND_DISK)
                
                # saving files
                filename_combinations = os.path.join(output_dir, 'files-%s-data' % key)
                if not cluster_execution:
                    filename_combinations = 'file://' + filename_combinations
                query_candidate_datasets_tmp.saveAsPickleFile(filename_combinations)

                query_candidate_datasets = sc.union([
                    query_candidate_datasets,
                    query_candidate_datasets_tmp
                ]).persist(StorageLevel.MEMORY_AND_DISK)

            # saving mapping
            filename_datasets = os.path.join(output_dir, 'id-to-dataset')
            if not cluster_execution:
                filename_datasets = 'file://' + filename_datasets
            dataset_id_to_data.saveAsPickleFile(filename_datasets)

    else:

        # datasets previously generated
        for key in ['training', 'testing']:
            filename_combinations = os.path.join(output_dir, 'files-%s-data' % key)
            if not cluster_execution:
                filename_combinations = 'file://' + filename_combinations
            query_candidate_datasets = sc.union([
                query_candidate_datasets,
                sc.pickleFile(filename_combinations)
            ]).persist(StorageLevel.MEMORY_AND_DISK)

        filename_datasets = os.path.join(output_dir, 'id-to-dataset')
        if not cluster_execution:
            filename_datasets = 'file://' + filename_datasets
        dataset_id_to_data = sc.pickleFile(filename_datasets).persist(StorageLevel.MEMORY_AND_DISK)

    if not skip_training_data:

        if not skip_training_data and skip_dataset_creation:
            start_time = time.time()

        if not query_candidate_datasets.isEmpty():

            # getting performance scores
            performance_scores = query_candidate_datasets.map(
                # first, let's use query dataset id as key
                lambda x: (x[0], (x[1], x[2]))
            ).join(
                # we get the query datasets
                dataset_id_to_data
            ).map(
                # (query dataset id, query dataset, target variable name, candidate dataset ids)
                lambda x: (x[0], x[1][1], x[1][0][0], x[1][0][1])
            ).flatMap(
                # then let's use each candidate dataset id as key
                lambda x: [(x[3][i], (x[0], x[1], x[2])) for i in range(len(x[3]))]
            ).join(
                # we get the candidate datasets
                dataset_id_to_data
            ).map(
                # ((query dataset id, query dataset, target variable name), ([candidate dataset id], [candidate dataset]))
                lambda x: ((x[1][0][0], x[1][0][1], x[1][0][2]), ([x[0]], [x[1][1]]))
            ).reduceByKey(
                # concatenating lists of candidate datasets
                lambda x, y: (x[0] + y[0], x[1] + y[1])
            ).flatMap(
                lambda x: generate_performance_scores(x[0][0], x[0][1], x[0][2], x[1][0], x[1][1], params)
            )

            # saving scores
            algorithm_name = params['regression_algorithm']
            if params['regression_algorithm'] == 'random forest':
                algorithm_name = 'random-forest'
            filename = os.path.join(output_dir, 'training-data-' + algorithm_name)
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

    if not skip_dataset_creation:
        print(' -- N. positive examples: %d' %n_positive_examples)
        print(' -- N. negative examples: %d' %n_negative_examples)
        print(' -- N. training datasets: %d' %n_training_datasets)
        print(' -- N. test datasets: %d' %n_test_datasets)
        print(' -- Processed datasets: %d' %processed_datasets.value)
        print(' -- Datasets w/o appropriate files: %d' %no_appropriate_files.value)
        print(' -- Datasets w/ no regression problem: %d' %no_regression.value)
        print(' -- Datasets w/ multiple data files: %d' %multiple_files.value)
        print(' -- Datasets w/o numeric targets: %d' %no_numerical_target.value)
        print(' -- Datasets w/o enough columns: %d' %no_enough_columns.value)
        print(' -- Datasets w/o enough records: %d' %no_enough_records.value)
        print(' -- Datasets w/ more than 900,000 records: %d' %many_records.value)
        print(' -- Datasets w/ pandas.Dataframe exception: %d' %dataframe_exception.value)
