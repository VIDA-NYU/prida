import copy
import csv
import pandas as pd
import multiprocessing
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, \
    mean_squared_log_error, median_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import shutil
import json

def train_and_test_model(data, target_variable_name):
    """Builds a model using data to predict the target variable.
    """

    X_train, X_test, y_train, y_test = train_test_split(
        data.drop(target_variable_name, axis=1),
        data[target_variable_name],
        test_size=0.33,
        random_state=42
    )

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

    return dict(
        mean_absolute_error=mean_absolute_error(y_test, yfit),
        mean_squared_error=mean_squared_error(y_test, yfit),
        median_absolute_error=median_absolute_error(y_test, yfit),
        r2_score=r2_score(y_test, yfit)
    )


def get_performance_scores(data, target_variable_name, missing_value_imputation):
    """Builds a model using data to predict the target variable,
    returning different performance metrics.
    """

    if missing_value_imputation:
        
        # imputation on data
        fill_NaN = SimpleImputer(missing_values=np.nan, strategy='mean')
        new_data = pd.DataFrame(fill_NaN.fit_transform(data))
        new_data.columns = data.columns
        new_data.index = data.index

        # training and testing model
        return train_and_test_model(new_data, target_variable_name)

    else:
        return train_and_test_model(data, target_variable_name)


def break_companion_and_join_datasets_per_record(record, dir_):

    output = list()

    test_record = json.loads(record)
        
    query_dataset = test_record['query_dataset']
    query_key = test_record['query_key']
    target = test_record['target']
    candidate_dataset = test_record['candidate_dataset']
    candidate_key = test_record['candidate_key']
    joined_dataset = test_record['joined_dataset']
    imputation_strategy = test_record['imputation_strategy']
    mean_absolute_error = test_record['mean_absolute_error']
    mean_squared_error = test_record['mean_squared_error']
    median_absolute_error = test_record['median_absolute_error']
    r2_score = test_record['r2_score']
    
    # reading query data
    query_data = pd.read_csv(query_dataset)
    
    # reading candidate data
    candidate_data = pd.read_csv(candidate_dataset)
    candidate_data = candidate_data.select_dtypes(exclude=['bool'])
    
    if len(candidate_data.columns) < 2:
        continue
    
    # reading joined dataset
    joined_data = pd.read_csv(joined_dataset)
    joined_data = joined_data.select_dtypes(exclude=['bool'])
    
    for column in candidate_data.columns:
        if column == candidate_key:
            continue
            
        # new test record
        new_record = copy.deepcopy(test_record)
        
        # creating new candidate dataset
        columns_to_drop = set(list(candidate_data.columns)).difference(
            set([candidate_key, column])
        )
        single_column_data = candidate_data.drop(
            list(columns_to_drop),
            axis=1
        )
        candidate_path = os.path.join(
            'companion-datasets-single-column',
            '%s_%s'%(
                os.path.basename(candidate_dataset),
                column.replace('%s'%os.path.sep, '_').strip()
            )
        )
        
        # creating new join dataset
        columns_to_drop = set(list(joined_data.columns)).difference(
            set(list(query_data.columns))
        ).difference(set([column]))
        single_column_joined_data = joined_data.drop(
            list(columns_to_drop),
            axis=1
        )
        join_path = os.path.join(
            dir_,
            '%s_%s.csv'%(
                os.path.splitext(os.path.basename(joined_dataset))[0],
                column.replace('%s'%os.path.sep, '_').strip()
            )
        )
        
        if single_column_joined_data.shape[1] == query_data.shape[1]:
            continue  # no join was performed
        
        # saving datasets
        if not os.path.exists(candidate_path):
            try:
                lock.acquire()
                single_column_data.to_csv(candidate_path, index=False)
            except:
                continue
            finally:
                lock.release()
        new_record['candidate_dataset'] = os.path.abspath(candidate_path)
        if not os.path.exists(join_path):
            try:
                lock.acquire()
                single_column_joined_data.to_csv(join_path, index=False)
            except:
                continue
            finally:
                lock.release()
        new_record['joined_dataset'] = os.path.abspath(join_path)

        # scores after augmentation
        scores_query_candidate = get_performance_scores(
            single_column_joined_data.drop([query_key], axis=1),
            target,
            True
        )
        
        new_record['mean_absolute_error'] = [mean_absolute_error[0],
                                             scores_query_candidate['mean_absolute_error']]
        new_record['mean_squared_error'] = [mean_squared_error[0],
                                           scores_query_candidate['mean_squared_error']]
        new_record['median_absolute_error'] = [median_absolute_error[0],
                                               scores_query_candidate['median_absolute_error']]
        new_record['r2_score'] = [r2_score[0],
                                  scores_query_candidate['r2_score']]

        output.append(json.dumps(new_record))

    return output


def pool_init(l):
    global lock
    lock = l


def break_companion_and_join_datasets(path_to_datamart_records, dir_):
    
    records = open(path_to_datamart_records).readlines()
    l = multiprocessing.Lock()
    p = multiprocessing.Pool(initializer=pool_init, initargs=(l,) processes=multiprocessing.cpu_count())
    new_records = p.starmap(
        break_companion_and_join_datasets_per_record,
        [(record, dir_) for record in records]
    )
            
    return new_records


# creating directories
if not os.path.exists('companion-datasets-single-column'):
    os.mkdir('companion-datasets-single-column')
for p in ['taxi-vehicle-collision', 'ny-taxi-demand', 'college-debt', 'poverty-estimation']:
    if not os.path.exists('companion-datasets-single-column/%s'%p):
        os.mkdir('companion-datasets-single-column/%s'%p)

## NY Taxi and Vehicle Collision Problem

taxi_records = break_companion_and_join_datasets(
    'taxi-vehicle-collision-datamart-records/datamart-records',
    'companion-datasets-single-column/taxi-vehicle-collision/'
)

print(len(taxi_records))
print(len(taxi_records[0]))

# if os.path.exists('taxi-vehicle-collision-datamart-records-single-column/'):
#     shutil.rmtree('taxi-vehicle-collision-datamart-records-single-column/')
# os.mkdir('taxi-vehicle-collision-datamart-records-single-column/')

# training_records = open('taxi-vehicle-collision-datamart-records-single-column/datamart-records', 'w')
# for record in taxi_records:
#     training_records.write(record + "\n")
# training_records.close()

# ## College Debt Problem

# college_debt_records = break_companion_and_join_datasets(
#     'college-debt-datamart-records/datamart-records',
#     'companion-datasets-single-column/college-debt/'
# )

# if os.path.exists('college-debt-datamart-records-single-column/'):
#     shutil.rmtree('college-debt-datamart-records-single-column/')
# os.mkdir('college-debt-datamart-records-single-column/')

# training_records = open('college-debt-datamart-records-single-column/datamart-records', 'w')
# for record in college_debt_records:
#     training_records.write(record + "\n")
# training_records.close()

# ## Poverty Estimation

# poverty_estimation_records = break_companion_and_join_datasets(
#     'poverty-estimation-datamart-records/datamart-records',
#     'companion-datasets-single-column/poverty-estimation/'
# )

# if os.path.exists('poverty-estimation-datamart-records-single-column/'):
#     shutil.rmtree('poverty-estimation-datamart-records-single-column/')
# os.mkdir('poverty-estimation-datamart-records-single-column/')

# training_records = open('poverty-estimation-datamart-records-single-column/datamart-records', 'w')
# for record in poverty_estimation_records:
#     training_records.write(record + "\n")
# training_records.close()