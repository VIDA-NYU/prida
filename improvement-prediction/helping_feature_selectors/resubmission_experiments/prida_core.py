'''
Here, we implement functions that are fundamental for all
versions of PRIDA (classic and hierarchical)
'''

import pandas as pd
import numpy as np
import os
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
import warnings; warnings.simplefilter('ignore')
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from prida_constants import *
import sys
sys.path.append('../../')
from feature_factory import *

def train_random_forest_regressor(features, gains):
    '''
    Builds a model using features to predict associated gains in R2-score
    '''
    feature_scaler = MinMaxScaler().fit(features)
    features_train = feature_scaler.transform(features)
    reg = RandomForestRegressor(n_estimators=100, random_state=42)
    reg.fit(features_train, gains)
    return feature_scaler, reg

def train_random_forest(features, classes):
    '''
    Builds a model using features to predict associated classes
    '''
    #print('using random forest')
    feature_scaler = MinMaxScaler().fit(features)
    features_train = feature_scaler.transform(features)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(features_train, classes)
    return feature_scaler, clf

def join_datasets(base_dataset,
                  candidate_datasets, 
                  base_key, 
                  mean_data_imputation=MEAN_DATA_IMPUTATION, 
                  #rename_numerical=RENAME_NUMERICAL, 
                  #separator=SEPARATOR, 
                  candidate_key_columns=None):
    '''
    Given (1) a base dataset, (2) candidate datasets with only two 
    columns (one key and one numerical attribute), and (3) keys  for joining purposes, 
    this function generates a big table composed of all joined datasets.
    '''

    augmented_dataset = base_dataset
    augmented_dataset.set_index(base_key, inplace=True)
    names_and_columns = {}
    for name in candidate_datasets.keys():
        try:
            if candidate_key_columns:
                augmented_dataset = pd.merge(augmented_dataset,
                                             candidate_datasets[name], 
                                             how='left',
                                             left_on=[base_key],
                                             right_on=[candidate_key_columns[name]],
                                             suffixes=['','_r'])
                names_and_columns[name] = list(set(candidate_datasets[name].columns.tolist()) - set([candidate_key_columns[name]]))
            else:
                augmented_dataset = pd.merge(augmented_dataset, 
                                             candidate_datasets[name], #.set_index(base_key, inplace=True),
                                             how='left',
                                             on=base_key,
                                             validate='m:1',
                                             suffixes=['','_r'])
                #print(name, list(candidate_datasets[name].columns))
                names_and_columns[name] = list(set(candidate_datasets[name].columns.tolist()) - set([base_key]))
                #print('done joining dataset', name)
        except (pd.errors.EmptyDataError, KeyError, ValueError) as e:
            print('there was an error for dataset', name, e)
            continue
    
    augmented_dataset = augmented_dataset.select_dtypes(include=['int64', 'float64'])
    augmented_dataset = augmented_dataset.replace([np.inf, -np.inf], np.nan)
    #print('augmented data shape', augmented_dataset.shape)
    if mean_data_imputation:
        mean = augmented_dataset.mean().replace(np.nan, 0.0)
        new_data = augmented_dataset.fillna(mean)
        new_data.index = augmented_dataset.index
        new_data.columns = augmented_dataset.columns
        print('number of initial features', len(base_dataset.columns), 'augmented dataset', len(augmented_dataset.columns))
        #print('kept', augmented_dataset.columns.tolist())
        print('leaving join datasets')
        new_data = new_data.loc[:,~new_data.columns.duplicated()]
        return new_data, names_and_columns

    #print('kept', augmented_dataset.columns.tolist())
    print('leaving join datasets')
    augmented_dataset = augmented_dataset.loc[:,~augmented_dataset.columns.duplicated()]
    return augmented_dataset, names_and_columns

def read_candidates(candidate_directory, base_key, separator=SEPARATOR, rename_numerical=RENAME_NUMERICAL):
    '''
    Given a directory with candidates, this function reads and partially 
    processes them
    '''
    candidates = {}
    candidate_names = [f for f in os.listdir(candidate_directory)]
    for name in candidate_names:
        dataset = pd.read_csv(os.path.join(candidate_directory, name), sep=separator)
        dataset = dataset.replace([np.inf, -np.inf], np.nan).dropna(how="all")
        dataset = dataset.replace([np.nan], 0.0).dropna(how="all")
        ### optional step:  rename the numerical column in the dataset
        if rename_numerical:
            numerical_column = [i for i in dataset.columns if i != base_key][0]
            dataset = dataset.rename(columns={numerical_column: name.split('.')[0]})
        candidates[name] = dataset
    return candidates

def compute_candidate_features(candidate_dataset, key):
    '''
    This function calculates the individual candidate features 
    (see CANDIDATE_FEATURES)
    '''

    candidate_dataset = candidate_dataset.set_index(key).fillna(candidate_dataset.mean())
    feature_factory_candidate = FeatureFactory(candidate_dataset)
    individual_features = feature_factory_candidate.get_individual_features(func=max_in_modulus)
    return individual_features

def normalize_features(features, scaler=None):
    '''
    This function normalizes features using sklearn's StandardScaler
    '''
    if not scaler:
        scaler = MinMaxScaler().fit(features)
    return scaler.transform(features)

def compute_complex_candidate_features(query_key_values,
                                       candidate_columns, 
                                       key, 
                                       target_name, 
                                       augmented_dataset):
    '''
    This function generates candidate-target and candidate-candidate features required to determine, 
    through classification, whether an augmentation with the candidate_dataset (which is single-feature) 
    is likely to hamper the model (or simply bring no gain)
    '''

    candidate_dataset = augmented_dataset[candidate_columns] 
    # Get candidate-target features
    ## The features are, in order: max_query_candidate_pearson, max_query_candidate_spearman, 
    ## max_query_candidate_covariance, max_query_candidate_mutual_info
    column_names = candidate_dataset.columns.tolist() + [target_name]
    feature_factory_candidate_target = FeatureFactory(augmented_dataset[column_names].fillna(augmented_dataset[column_names].mean()))
    candidate_features_target = feature_factory_candidate_target.get_pairwise_features_with_target(target_name, func=max_in_modulus)
    # Get query-candidate feature "containment ratio". 
    candidate_key_values = candidate_dataset.index.values
    intersection_size = len(set(query_key_values) & set(candidate_key_values))
    containment_ratio = [intersection_size/len(query_key_values)]
    return candidate_features_target, containment_ratio
    
