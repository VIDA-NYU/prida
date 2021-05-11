'''
Here, we implement baseline pruners such as the classic version of PRIDA, a 
regression-based version, and a containment-based method.
'''

import pandas as pd
import numpy as np
import time
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
import warnings; warnings.simplefilter('ignore')
from sklearn.ensemble import RandomForestClassifier
from prida_constants import *
import sys
sys.path.append('../../')
from feature_factory import *
from prida_core import *

def prune_candidates_classic(training_data,  
                             base_dataset,
                             candidate_directory,
                             key, 
                             target_name,
                             topN=100):
    '''
    This function trains and uses the classic PRIDA version (non-hierarichical) 
    as a pruner of candidates for augmentation.
    It keeps the topN candidates.
    '''
    
    #Let's train the first, dataset-feature-based model over the training dataset
    time1 = time.time()
    feature_scaler, model = train_random_forest(training_data[FEATURES], training_data[CLASS_ATTRIBUTE_NAME]) 
    time2 = time.time()
    print('time to train model', (time2-time1)*1000.0, 'ms')
    
    #Generate a label for every feature in the augmented dataset 
    time1 = time.time()
    
    ## We just need to compute query and query-target features once!
    ## In order, the returned features are number_of_columns, number_of_rows, row_to_column_ratio,
    ## max_mean, max_outlier_percentage, max_skewness, max_kurtosis, max_number_of_unique_values.
    feature_factory_query = FeatureFactory(base_dataset.drop([target_name], axis=1))
    query_features = feature_factory_query.get_individual_features(func=max_in_modulus)

    ## The features are, in order: max_query_target_pearson, max_query_target_spearman, 
    ## max_query_target_covariance, max_query_target_mutual_info
    feature_factory_full_query = FeatureFactory(base_dataset)
    query_features_target = feature_factory_full_query.get_pairwise_features_with_target(target_name, 
                                                                                         func=max_in_modulus)

    # Now let's create a single table once, i.e., perform all the necessary inner joins at once
    candidates = read_candidates(candidate_directory, key)
    augmented_dataset, names_and_columns = join_datasets(base_dataset, candidates, key)
    query_key_values = base_dataset.index.values
    feature_vectors = []
    for name in sorted(candidates.keys()):
        candidate_features = compute_candidate_features(candidates[name], key)
        candidate_target_features, candidate_query_features = compute_complex_candidate_features(query_key_values,
                                                                                                 names_and_columns[name], 
                                                                                                 key, 
                                                                                                 target_name, 
                                                                                                 augmented_dataset)
        feature_vectors.append(query_features + candidate_features + query_features_target + candidate_target_features + candidate_query_features)
    predictions = model.predict(normalize_features(np.array(feature_vectors))) 
    gain_pred_probas = [elem[0] for elem in model.predict_proba(normalize_features(np.array(feature_vectors)))]
    probs_dictionary = {names_and_columns[name][0]: prob for name, prob in zip(sorted(candidates.keys()), list(gain_pred_probas))}
    pruned = sorted(probs_dictionary.items(), key = lambda x:x[1], reverse=True)[:topN]
        
    candidates_to_keep = [elem[0] for elem in pruned if elem[1] > 0.5] # if elem[1] > 0.5, it was classified as 'keepable'
    time2 = time.time()
    print('time to predict what candidates to keep', (time2-time1)*1000.0, 'ms')
    print('initial number of candidates', len(candidates.keys()), 'final number of candidates', len(candidates_to_keep))
    return augmented_dataset[base_dataset.columns.to_list() + candidates_to_keep]    

def prune_containment_based(base_dataset, 
                            dataset_directory, 
                            base_key,
                            topN=100, 
                            mean_data_imputation=MEAN_DATA_IMPUTATION,
                            rename_numerical=RENAME_NUMERICAL, 
                            separator=SEPARATOR, 
                            candidate_key_columns=None):
    '''
    Given (1) a base dataset, (2) a directory with datasets that only have two 
    columns (one key and one numerical attribute), and (3) keys  for joining purposes,  
    this function generates an augmented table composed of the original base attributes and 
    the topN overlapping candidates (higher containment). 
    '''

    time1 = time.time()
    augmented_dataset = base_dataset
    #augmented_dataset.set_index(base_key, inplace=True)
    dataset_names = [f for f in os.listdir(dataset_directory)]
    
    containments = {}
    for name in dataset_names:
        try:
            ### Step 1: read the dataset in the directory
            dataset = pd.read_csv(os.path.join(dataset_directory, name), 
                                  sep=separator)
            dataset = dataset.replace([np.inf, -np.inf], np.nan).dropna(how="all")
            dataset = dataset.replace([np.nan], 0.0).dropna(how="all")
            
            ### Step 2 (optional):  rename the numerical column in the dataset
            if rename_numerical:
                numerical_column = [i for i in dataset.columns if i != base_key][0]
                dataset = dataset.rename(columns={numerical_column: name.split('.')[0]})

            ### Step 3 (compute the containment ratio and store it)
            base_keys = set(base_dataset[base_key])
            if candidate_key_columns:
                candidate_keys = set(dataset[candidate_key_columns[name]])
            else:
                candidate_keys = set(dataset[base_key])
            intersection_size = len(base_keys & candidate_keys)
            containment_ratio = intersection_size/len(base_keys)
            containments[name] = {'ratio': containment_ratio, 'candidate_dataset': dataset}
        except (pd.errors.EmptyDataError, KeyError, ValueError) as e:
            print('there was an error for dataset', name, e)
            continue

    chosen_candidates = [elem[0] for elem in sorted(containments.items(), key= lambda x: x[1]['ratio'], reverse=True)[:topN]]
    for name in chosen_candidates:
        try:
            dataset = containments[name]['candidate_dataset']
            if candidate_key_columns:
                augmented_dataset = pd.merge(augmented_dataset,
                                             dataset,
                                             how='left',
                                             left_on=[base_key],
                                             right_on=[candidate_key_columns[name]])
            else:
                dataset.set_index(base_key, inplace=True)
                augmented_dataset = augmented_dataset.join(dataset, how='left')
                
        except (pd.errors.EmptyDataError, KeyError, ValueError):
            continue
    
    augmented_dataset = augmented_dataset.select_dtypes(include=['int64', 'float64'])
    augmented_dataset = augmented_dataset.replace([np.inf, -np.inf], np.nan)
    if mean_data_imputation:
        mean = augmented_dataset.mean().replace(np.nan, 0.0)
        new_data = augmented_dataset.fillna(mean)
        new_data.index = augmented_dataset.index
        new_data.columns = augmented_dataset.columns
        time2 = time.time()
        print('time to perform join', (time2-time1)*1000.0, 'ms')
        print('number of initial features', len(base_dataset.columns), 'augmented dataset', len(augmented_dataset.columns))
        #print('kept', augmented_dataset.columns.tolist())
        print('leaving join datasets')
        new_data = new_data.loc[:,~new_data.columns.duplicated()]
        return new_data

    time2 = time.time()
    print('time to perform join', (time2-time1)*1000.0, 'ms')
    print('number of initial features', len(base_dataset.columns), 'augmented dataset', len(augmented_dataset.columns))
    #print('kept', augmented_dataset.columns.tolist())
    print('leaving join datasets')
    augmented_dataset = augmented_dataset.loc[:,~augmented_dataset.columns.duplicated()]
    return augmented_dataset

def prune_candidates_regression(training_data,  
                                base_dataset,
                                candidate_directory,
                                key, 
                                target_name,
                                topN=100):
    '''
    This function trains and uses a regression-based version of PRIDA  
    as a pruner of candidates for augmentation. It keeps the topN candidates.
    '''
    
    #Let's train the  model over the training dataset
    time1 = time.time()
    feature_scaler, model = train_random_forest_regressor(training_data[FEATURES], training_data[GAIN_ATTRIBUTE_NAME]) 
    time2 = time.time()
    print('time to train model', (time2-time1)*1000.0, 'ms')
    
    #Generate a label for every feature in the augmented dataset 
    time1 = time.time()
    
    ## We just need to compute query and query-target features once!
    ## In order, the returned features are number_of_columns, number_of_rows, row_to_column_ratio,
    ## max_mean, max_outlier_percentage, max_skewness, max_kurtosis, max_number_of_unique_values.
    feature_factory_query = FeatureFactory(base_dataset.drop([target_name], axis=1))
    query_features = feature_factory_query.get_individual_features(func=max_in_modulus)

    ## The features are, in order: max_query_target_pearson, max_query_target_spearman, 
    ## max_query_target_covariance, max_query_target_mutual_info
    feature_factory_full_query = FeatureFactory(base_dataset)
    query_features_target = feature_factory_full_query.get_pairwise_features_with_target(target_name, 
                                                                                         func=max_in_modulus)

    # Now let's create a single table once, i.e., perform all the necessary inner joins at once
    candidates = read_candidates(candidate_directory, key)
    augmented_dataset, names_and_columns = join_datasets(base_dataset, candidates, key)
    query_key_values = base_dataset.index.values
    feature_vectors = []
    for name in sorted(candidates.keys()):
        candidate_features = compute_candidate_features(candidates[name], key)
        candidate_target_features, candidate_query_features = compute_complex_candidate_features(query_key_values,
                                                                                                 names_and_columns[name], 
                                                                                                 key, 
                                                                                                 target_name, 
                                                                                                 augmented_dataset)
        feature_vectors.append(query_features + candidate_features + query_features_target + candidate_target_features + candidate_query_features)

    predictions = model.predict(normalize_features(np.array(feature_vectors))) 
    scores_dictionary = {names_and_columns[name][0]: score for name, score in zip(sorted(candidates.keys()), predictions)}
    pruned = sorted(scores_dictionary.items(), key = lambda x:x[1], reverse=True)[:topN]
        
    candidates_to_keep = [elem[0] for elem in pruned if elem[1] > 0.5] # if elem[1] > 0.5, it was classified as 'keepable'
    time2 = time.time()
    print('time to predict what candidates to keep', (time2-time1)*1000.0, 'ms')
    print('initial number of candidates', len(candidates.keys()), 'final number of candidates', len(candidates_to_keep))
    return augmented_dataset[base_dataset.columns.to_list() + candidates_to_keep]    
