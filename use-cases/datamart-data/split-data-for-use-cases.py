import copy
from hdfs import InsecureClient
from io import StringIO
import json
import math
import numpy as np
import os
import pandas as pd
from pyspark import SparkConf, SparkContext, StorageLevel
import re
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, \
    mean_squared_log_error, median_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sys
import uuid
import time
from xgboost import XGBRegressor

# regex to take care of XGBoost ValueError
regex = re.compile(r"\[|\]|<", re.IGNORECASE)


def create_dir(file_path, hdfs_client):
    """Creates a new directory specified by file_path.
    Returns True on success.
    """

    if hdfs_client.status(file_path, strict=False):
        hdfs_client.delete(file_path, recursive=True, skip_trash=True)
    hdfs_client.makedirs(file_path)
    return True


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

            # training and test model
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


def break_companion_and_join_datasets(query_dataset, candidate_dataset, candidate_column, joined_dataset, record):
        
    # record information
    query_data_name = record['query_dataset']
    query_key = record['query_key']
    target = record['target']
    candidate_data_name = record['candidate_dataset']
    candidate_key = record['candidate_key']
    joined_data_name = record['joined_dataset']
    mean_absolute_error = record['mean_absolute_error']
    mean_squared_error = record['mean_squared_error']
    median_absolute_error = record['median_absolute_error']
    r2_score = record['r2_score']

    # params
    algorithm = 'random forest'
    inner_join = False
    
    # reading query data
    query_data = pd.read_csv(StringIO(query_dataset))
    
    # reading candidate data
    candidate_data = pd.read_csv(StringIO(candidate_dataset))
    candidate_data = candidate_data.select_dtypes(exclude=['bool'])
    
    if len(candidate_data.columns) < 2:
        return list()
    
    # reading joined dataset
    joined_data = pd.read_csv(StringIO(joined_dataset))
    joined_data = joined_data.select_dtypes(exclude=['bool'])
    
    column = list(candidate_data.columns)[candidate_column]

    if column == candidate_key:
        return list()
    
    # creating new candidate dataset
    columns_to_drop = set(list(candidate_data.columns)).difference(
        set([candidate_key, column])
    )
    single_column_data = candidate_data.drop(
        list(columns_to_drop),
        axis=1
    )
    
    # creating new join dataset
    columns_to_drop = set(list(joined_data.columns)).difference(
        set(list(query_data.columns))
    ).difference(set([column]))
    single_column_joined_data = joined_data.drop(
        list(columns_to_drop),
        axis=1
    )
    
    if single_column_joined_data.shape[1] == query_data.shape[1]:
        return list()  # no join was performed

    # new test record
    new_record = copy.deepcopy(record)

    # scores after augmentation
    imputation_strategy, scores_after = get_performance_scores(
        single_column_joined_data.drop([query_key], axis=1),
        target,
        algorithm,
        not(inner_join)
    )
    
    new_record['mean_absolute_error'] = [mean_absolute_error[0],
                                         scores_after[0]]
    new_record['mean_squared_error'] = [mean_squared_error[0],
                                       scores_after[1]]
    new_record['median_absolute_error'] = [median_absolute_error[0],
                                           scores_after[2]]
    new_record['r2_score'] = [r2_score[0],
                              scores_after[3]]

    new_record['mark'] = 'n/a'
    
    return [(
        query_dataset,
        single_column_data.to_csv(index=False),
        single_column_joined_data.to_csv(index=False),
        new_record
    )]


def save_id_to_record(id_, record, key_name):

    record[key_name] = id_
    return record


if __name__ == '__main__':

    start_time = time.time()

    # Spark context
    conf = SparkConf().setAppName("Data Generation for Use Cases")
    sc = SparkContext(conf=conf)

    # parameters
    params = json.load(open(".params.json"))
    training_records_file = params['training_records']
    datasets_dir = params['datasets_directory']
    output_dir = params['new_datasets_directory']
    hdfs_address = params['hdfs_address']
    hdfs_user = params['hdfs_user']

    # HDFS Client
    hdfs_client = InsecureClient(hdfs_address, user=hdfs_user)

    create_dir(output_dir, hdfs_client)

    # query datasets
    filename = os.path.join(datasets_dir, 'query-datasets', '*')
    query_datasets = sc.wholeTextFiles(filename).map(
        lambda x: (os.path.basename(x[0]), x[1])
    )

    # companion datasets
    filename = os.path.join(datasets_dir, 'companion-datasets', '*')
    companion_datasets = sc.wholeTextFiles(filename).map(
        lambda x: (os.path.basename(x[0]), x[1])
    )

    # join datasets
    filename = os.path.join(datasets_dir, 'join-datasets', '*')
    join_datasets = sc.wholeTextFiles(filename).map(
        lambda x: (os.path.join(*(x[0].split(os.path.sep)[-2:])), x[1])
    )

    # reading records and retrieving new data
    training_records = sc.textFile(training_records_file).repartition(1000).map(
        lambda x: json.loads(x)
    ).map(
        # (query dataset name, record)
        lambda x: (os.path.basename(x['query_dataset']), x)
    ).join(query_datasets).map(
        # (companion dataset name, (query dataset, record))
        lambda x: (os.path.basename(x[1][0]['candidate_dataset']), (x[1][1], x[1][0]))
    ).join(companion_datasets).map(
        # (join dataset name, (query dataset, companion dataset, record))
        lambda x: (os.path.join(*(x[1][0][1]['joined_dataset'].split(os.path.sep)[-2:])), (x[1][0][0], x[1][1], x[1][0][1]))
    ).join(join_datasets).map(
        # (query dataset, companion dataset, join dataset, record)
        lambda x: (x[1][0][0], x[1][0][1], x[1][1], x[1][0][2])
    ).flatMap(
        # (query dataset, candidate dataset, candidate column, joined dataset, record)
        lambda x: [(x[0], x[1], i, x[2], x[3]) for i in range(len(pd.read_csv(StringIO(x[1])).select_dtypes(exclude=['bool']).columns))]
    ).flatMap(
        # (query dataset, candidate dataset, joined dataset, record)
        lambda x: break_companion_and_join_datasets(x[0], x[1], x[2], x[3], x[4])
    ).persist(StorageLevel.MEMORY_AND_DISK)

    # getting ids for query datasets

    dataset_id_to_data_query = training_records.map(
        lambda x: x[0]
    ).distinct().map(
        lambda x: (str(uuid.uuid4()), x)
    ).persist(StorageLevel.MEMORY_AND_DISK)

    id_query_training_records = training_records.map(
        # key => query dataset
        lambda x: (x[0], (x[1], x[2], x[3]))
    ).join(
        dataset_id_to_data_query.map(lambda x: (x[1], x[0])), numPartitions=1000
    ).map(
        # replacing query dataset name for id inside the records
        lambda x: (x[1][0][0], x[1][0][1], save_id_to_record(x[1][1], x[1][0][2], 'query_dataset'))
    ).persist(StorageLevel.MEMORY_AND_DISK)

    # getting ids for candidate datasets

    print("Distinct: %d"%(id_query_training_records.map(lambda x: x[0]).distinct().count()))

    id_candidate_training_records = id_query_training_records.map(
        # key => candidate dataset
        lambda x: (x[0], [(x[1], x[2])])
    ).reduceByKey(
        lambda x, y: x + y
    ).map(
        # generating id
        # ((candidate datase id, candidate dataset), list of (joined dataset, record))
        lambda x: ((str(uuid.uuid4()), x[0]), x[1])
    ).map(
        # replacing candidate dataset name for id inside the records
        lambda x: (x[0], [(elem[0], save_id_to_record(x[0][0], elem[1], 'candidate_dataset')) for elem in x[1]])
    ).persist(StorageLevel.MEMORY_AND_DISK)

    dataset_id_to_data_candidate = id_candidate_training_records.map(
        lambda x: x[0]
    ).persist(StorageLevel.MEMORY_AND_DISK)

    # getting ids for joined datasets

    id_joined_training_records = id_candidate_training_records.flatMap(
        lambda x: x[1]
    ).repartition(1000).map(
        # key => joined dataset
        lambda x: (x[0], [x[1]])
    ).reduceByKey(
        lambda x, y: x + y
    ).map(
        # generating id
        # ((joined datase id, joined dataset), list of (record))
        lambda x: ((str(uuid.uuid4()), x[0]), x[1])
    ).map(
        # replacing joined dataset name for id inside the records
        lambda x: (x[0], [save_id_to_record(x[0][0], elem, 'joined_dataset') for elem in x[1]])
    ).persist(StorageLevel.MEMORY_AND_DISK)

    dataset_id_to_data_joined = id_joined_training_records.map(
        lambda x: x[0]
    ).persist(StorageLevel.MEMORY_AND_DISK)

    # training records
    all_training_records = id_joined_training_records.flatMap(
        lambda x: x[1]
    ).persist(StorageLevel.MEMORY_AND_DISK)

    # id to dataset mapping
    #   format is the following:
    #   (dataset id, dataset)
    dataset_id_to_data = sc.union([
        dataset_id_to_data_query,
        dataset_id_to_data_candidate,
        dataset_id_to_data_joined
    ]).persist(StorageLevel.MEMORY_AND_DISK)

    # name of the ml algorithm used to generate the performance scores
    algorithm_name = 'random-forest'

    # saving files
    filename = os.path.join(output_dir, 'training-data-%s' % algorithm_name)
    all_training_records.map(
        lambda x: json.dumps(x)
    ).repartition(1000).saveAsTextFile(filename)

    filename = os.path.join(output_dir, 'id-to-dataset-training')
    dataset_id_to_data.repartition(1000).saveAsPickleFile(filename)

    print('Duration: %.4f seconds' % (time.time() - start_time))
    print(' -- Configuration:')
    print('    . datasets_directory: %s' % params['datasets_directory'])
    print('    . new_datasets_directory: %s' % params['new_datasets_directory'])
    print('    . hdfs_address: %s' % params['hdfs_address'])
    print('    . hdfs_user: %s' % params['hdfs_user'])
    print('    . training_records: %s' % str(params['training_records']))
