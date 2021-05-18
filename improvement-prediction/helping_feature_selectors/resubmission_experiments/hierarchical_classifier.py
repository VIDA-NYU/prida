'''
A VLDB reviewer proposed the following: compute the features for each dataset in isolation (since it looks that few of them are quite predictive from Figure 4) 
and do early pruning of candidates before computing the full feature set. It's like a hierarchical classifier. 

Concretely, given a base table, a path to candidates, the name of the key and of the target, this script shows efficiency and effectiveness values for different 
percentages of pruning USING A HIERARCHICAL CLASSIFIER THAT AVOIDS PERFORMING JOINS UNTIL CLOSE TO THE END. 

argv[1] => base table
argv[2] => candidates' directory
argv[3] => key
argv[4] => target variable
argv[5] => feature selector (e.g., rifs or recursive_feature_elimination)
argv[6] => prepruner (e.g., prune_candidates_hierarchical, prune_candidates_classic, prune_containment_based, or prune_candidates_regression)
'''

import sys
import pandas as pd
import numpy as np
import time
import os
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import warnings; warnings.simplefilter('ignore')
from sklearn.ensemble import RandomForestClassifier
sys.path.append('../../')
from feature_factory import *
from feature_selectors import *
from prida_constants import *
from baseline_pruners import *
from prida_core import *

def compute_user_model_performance(dataset, target_name, attributes, model_type='random_forest'):
    '''
    This function checks how well a model (assumed to be the user's model), 
    trained on a given set of attributes, performs in the prediction of a target
    '''
    print('**** ATTRIBUTES', list(attributes))
    time1 = time.time()
    # Now let's split the data
    #dataset.dropna(inplace=True)
    mean = dataset.mean().replace(np.nan, 0.0)
    dataset = dataset.fillna(mean)
    #indices_to_keep = ~dataset.isin([np.nan, np.inf, -np.inf]).any(1)
    #dataset = dataset[indices_to_keep]#.astype(np.float64)
    #print(dataset.shape)
    X_train, X_test, y_train, y_test = train_test_split(dataset[attributes], 
                                                        dataset[target_name],
                                                        test_size=0.33,
                                                        random_state=42)
    if model_type == 'random_forest':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train[attributes], y_train.ravel())
        y_pred = model.predict(X_test[attributes])
    elif model_type == 'linear_regression':
        model = LinearRegression()
        model.fit(X_train[attributes], y_train.ravel())
        y_pred = model.predict(X_test[attributes])
    else:
        print('Specified user model is not implemented')
        exit()
    time2 = time.time()
    print('time to create user\'s model with chosen candidates', (time2-time1)*1000.0, 'ms')
    print('R2-score of user model', r2_score(y_test, y_pred))

def prune_candidates_hierarchical(training_data,  
                                  base_dataset,
                                  candidate_directory,
                                  key, 
                                  target_name,
                                  topN=100):
    '''
    This function trains and uses a hierarchical classifier as a pruner of candidates for augmentation.
    It keeps the topN candidates.
    '''
    
    #Let's train the first, dataset-feature-based model over the training dataset
    time1 = time.time()
    feature_scaler, model1 = train_random_forest(training_data[DATASET_FEATURES + QUERY_TARGET_FEATURES], training_data[CLASS_ATTRIBUTE_NAME]) 
    time2 = time.time()
    print('time to train dataset-feature/query-target-feature model', (time2-time1)*1000.0, 'ms')
    
    #Generate a label for every feature in the augmented dataset according to model1
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
    
    candidates = read_candidates(candidate_directory, key)
    individual_candidate_features = {}
    feature_vectors = []
    for name in sorted(candidates.keys()):
        candidate_features = compute_candidate_features(candidates[name], key)
        individual_candidate_features[name] = candidate_features
        feature_vectors.append(query_features + candidate_features + query_features_target)
        
    predictions1 = model1.predict(normalize_features(np.array(feature_vectors)))
    candidates_kept = {name: candidates[name] for name, pred in zip(sorted(candidates.keys()), predictions1) if pred == 'gain'}
    time2 = time.time()
    print('time to predict what candidates to keep with model1', (time2-time1)*1000.0, 'ms')

    #Let's train the second, full-feature model over the training dataset
    time1 = time.time()
    feature_scaler, model2 = train_random_forest(training_data[DATASET_FEATURES + QUERY_TARGET_FEATURES + CANDIDATE_TARGET_FEATURES + DATASET_DATASET_FEATURES],
                                                 training_data[CLASS_ATTRIBUTE_NAME]) 
    time2 = time.time()
    print('time to train full-feature model', (time2-time1)*1000.0, 'ms')

    #Let's augment the dataset with all candidates that were kept by model1
    time1 = time.time()
    augmented_dataset, names_and_columns = join_datasets(base_dataset, candidates_kept, key)
    time2 = time.time()
    print('time to augment dataset with candidates kept by model1', (time2-time1)*1000.0, 'ms')

    #Now let's generate a label for every feature in the augmented dataset according to model2
    time1 = time.time()
    query_key_values = base_dataset.index.values
    feature_vectors = []
    for name in sorted(candidates_kept.keys()):
        candidate_target_features, candidate_query_features = compute_complex_candidate_features(query_key_values,
                                                                                                 names_and_columns[name], 
                                                                                                 key, 
                                                                                                 target_name, 
                                                                                                 augmented_dataset)
        if candidate_target_features and candidate_query_features:
            feature_vectors.append(query_features + individual_candidate_features[name] + query_features_target + candidate_target_features + candidate_query_features)

    if feature_vectors:
        predictions2 = model2.predict(normalize_features(np.array(feature_vectors)))
        gain_pred_probas = [elem[0] for elem in model2.predict_proba(normalize_features(np.array(feature_vectors)))]
        probs_dictionary = {name: prob for name, prob in zip(sorted(candidates_kept.keys()), list(gain_pred_probas))}
        pruned = sorted(probs_dictionary.items(), key = lambda x:x[1], reverse=True)[:topN]
        
        final_kept_candidates = {elem[0]: candidates[elem[0]] for elem in pruned if elem[1] > 0.5}
        
        #[elem[0] for elem in pruned if elem[1] > 0.5] # if elem[1] > 0.5, it was classified as 'keepable'
        time2 = time.time()
        print('time to predict what candidates to keep', (time2-time1)*1000.0, 'ms')
        print('initial number of candidates', len(candidates.keys()),
              'mid number of candidates', len(candidates_kept.keys()),
              'final number of candidates', len(final_kept_candidates))
        augmented_dataset, names_and_columns = join_datasets(base_dataset.reset_index(), final_kept_candidates, key)
        return augmented_dataset
    return base_dataset

def check_efficiency_and_effectiveness(base_dataset,
                                       path_to_candidates,
                                       key, 
                                       target,
                                       training_data,
                                       rename_numerical=True,
                                       separator=SEPARATOR,
                                       feature_selector=rifs,
                                       prepruning=prune_candidates_hierarchical, 
                                       topN=100):
    '''
    This function gets the time to run a feature selector using 
    different prepruners
    '''
    
    print('Initial performance')
    compute_user_model_performance(base_dataset, target, base_dataset.drop([key, target], axis=1).columns)
    print('******* PRUNING ********')
    #Step 2: let's see how much time it takes to run the classifier-based pruner
    if prepruning == 'prune_candidates_hierarchical':
        augmented_dataset = prune_candidates_hierarchical(training_data,  
                                                          base_dataset,
                                                          path_to_candidates,
                                                          key, 
                                                          target,
                                                          topN=topN)

        print('candidates kept by hierarchical classifier', augmented_dataset.columns.to_list())

    elif prepruning == 'prune_candidates_classic':
        augmented_dataset = prune_candidates_classic(training_data,  
                                                     base_dataset,
                                                     path_to_candidates,
                                                     key, 
                                                     target,
                                                     topN=topN)
        print('candidates kept by classic PRIDA classifier', augmented_dataset.columns.to_list())

    elif prepruning == 'prune_containment_based':
        augmented_dataset = prune_containment_based(base_dataset,
                                                    path_to_candidates,
                                                    key,
                                                    topN=topN)
        print('candidates kept by containment strategy', augmented_dataset.columns.to_list())

    elif prepruning == 'prune_candidates_regression':
        augmented_dataset = prune_candidates_regression(training_data,  
                                                        base_dataset,
                                                        path_to_candidates,
                                                        key, 
                                                        target,
                                                        topN=topN)
        print('candidates kept by the regression-based version of PRIDA', augmented_dataset.columns.to_list())
    elif prepruning == 'prune_candidates_hybrid':
        augmented_dataset = prune_candidates_hybrid(training_data,  
                                                    base_dataset,
                                                    path_to_candidates,
                                                    key, 
                                                    target,
                                                    topN=topN)
        print('candidates kept by the hybrid (classification+regression) version of PRIDA', augmented_dataset.columns.to_list())
    else:
        print('prepruner that was passed is not implemented')
        exit()
        
    if sorted(augmented_dataset.columns.tolist()) == sorted(base_dataset.columns.tolist()):
        print('All candidates were pruned and no augmentation was performed')
        exit()
    #Step 3: select features with selector over pruned dataset (if RIFS, we inject 20% of random features)
    time1 = time.time()
    if feature_selector == 'rifs':
        selected_pruned = rifs(augmented_dataset,  
                               target, 
                               key) 
    elif feature_selector == 'recursive_feature_elimination':
        selected_pruned = recursive_feature_elimination(augmented_dataset.drop([target], axis=1), augmented_dataset[target])
    else:
        print('feature selector that was passed is not implemented')
        exit()
    time2 = time.time()
    print('time to run feature selector', (time2-time1)*1000.0, 'ms')

    #Step 4: compute the user's regression model with features selected_pruned 
    if len(selected_pruned) == 0:
        print('No features were selected. Can\'t run user\'s model.')
    else:
        time1 = time.time()
        compute_user_model_performance(augmented_dataset, 
                                       target, 
                                       selected_pruned)
        time2 = time.time()
        print('time to create and assess user\'s model with pruner', prepruning, (time2-time1)*1000.0, 'ms')

if __name__ == '__main__':    
    path_to_base_table = sys.argv[1]
    path_to_candidates = sys.argv[2]
    key = sys.argv[3]
    target = sys.argv[4]
    feature_selector = sys.argv[5]
    prepruning = sys.argv[6]
    
    openml_training = pd.read_csv(TRAINING_FILENAME)
    openml_training[CLASS_ATTRIBUTE_NAME] = ['gain' if row['gain_in_r2_score'] > 0 else 'loss'
                                        for index, row in openml_training.iterrows()]
    openml_training_high_containment = openml_training.loc[openml_training['containment_fraction'] >= THETA]

    base_table = pd.read_csv(path_to_base_table)

    check_efficiency_and_effectiveness(base_table,
                                       path_to_candidates,
                                       key, 
                                       target,
                                       openml_training_high_containment,
                                       rename_numerical=True,
                                       separator=SEPARATOR,
                                       topN=100,
                                       feature_selector=feature_selector,
                                       prepruning=prepruning)
