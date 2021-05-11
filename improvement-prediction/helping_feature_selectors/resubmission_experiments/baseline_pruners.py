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
    if not topN:
        pruned = sorted(probs_dictionary.items(), key = lambda x:x[1], reverse=True)[:int((1.0 - percentage)*len(probs_dictionary.items()))]
    else:
        pruned = sorted(probs_dictionary.items(), key = lambda x:x[1], reverse=True)[:topN]
        
    candidates_to_keep = [elem[0] for elem in pruned if elem[1] > 0.5] # if elem[1] > 0.5, it was classified as 'keepable'
    time2 = time.time()
    print('time to predict what candidates to keep', (time2-time1)*1000.0, 'ms')
    print('initial number of candidates', len(candidates.keys()), 'final number of candidates', len(candidates_to_keep))
    return augmented_dataset[base_dataset.columns.to_list() + candidates_to_keep]    

