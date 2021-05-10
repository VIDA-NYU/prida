'''
A VLDB reviewer proposed the following: compute the features for each dataset in isolation (since it looks that few of them are quite predictive from Figure 4) 
and do early pruning of candidates before computing the full feature set. It's like a hierarchical classifier. 

Concretely, given a base table, a path to candidates, the name of the key and of the target, this script shows efficiency and effectiveness values for different 
percentages of pruning USING A HIERARCHICAL CLASSIFIER THAT AVOIDS PERFORMING JOINS UNTIL CLOSE TO THE END. 

argv[1] => base table
argv[2] => candidates' directory
argv[3] => key
argv[4] => target variable
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
sys.path.append('.')
from feature_factory import *
from feature_selectors import *

TRAINING_FILENAME = '../../classification/training-simplified-data-generation.csv'
THETA = 0.0
SEPARATOR = ','
RENAME_NUMERICAL = True
MEAN_DATA_IMPUTATION = True

CLASS_ATTRIBUTE_NAME = 'class_pos_neg'

DATASET_FEATURES = ['query_num_of_columns', 'query_num_of_rows', 'query_row_column_ratio', 'query_max_mean', 'query_max_outlier_percentage', 
                    'query_max_skewness', 'query_max_kurtosis', 'query_max_unique', 'candidate_num_of_columns', 'candidate_num_rows', 
                    'candidate_row_column_ratio', 'candidate_max_mean', 'candidate_max_outlier_percentage', 'candidate_max_skewness',
                    'candidate_max_kurtosis', 'candidate_max_unique']
QUERY_TARGET_FEATURES = ['query_target_max_pearson', 'query_target_max_spearman', 'query_target_max_covariance', 'query_target_max_mutual_info']
CANDIDATE_TARGET_FEATURES = ['candidate_target_max_pearson', 'candidate_target_max_spearman',
                             'candidate_target_max_covariance', 'candidate_target_max_mutual_info']
DATASET_DATASET_FEATURES = ['containment_fraction']

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
                                             right_on=[candidate_key_columns[name]])
                names_and_columns[name] = list(set(candidate_datasets[name].columns.tolist()) - set([candidate_key_columns[name]]))
            else:
                augmented_dataset = pd.merge(augmented_dataset, 
                                             candidate_datasets[name], #.set_index(base_key, inplace=True),
                                             how='left',
                                             on=base_key,
                                             validate='m:1')
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
    candidate_dataset = augmented_dataset[candidate_columns] #name]].set_index(key).fillna(candidate_dataset.mean())
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

def prune_candidates_hierarchical(training_data,  
                                  base_dataset,
                                  candidate_directory,
                                  key, 
                                  target_name,
                                  topN=100):
    '''
    This function trains and uses a hierarchical classifier as a pruner of candidates for augmentation.
    It keeps the top percentage (indicated by parameter percentage) of candidates.
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
    print('time to train augment dataset with candidates kept by model1', (time2-time1)*1000.0, 'ms')

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
        feature_vectors.append(query_features + individual_candidate_features[name] + query_features_target + candidate_target_features + candidate_query_features)

    predictions2 = model2.predict(normalize_features(np.array(feature_vectors)))
    gain_pred_probas = [elem[0] for elem in model2.predict_proba(normalize_features(np.array(feature_vectors)))]
    probs_dictionary = {name: prob for name, prob in zip(sorted(candidates_kept.keys()), list(gain_pred_probas))}
    if not topN:
        pruned = sorted(probs_dictionary.items(), key = lambda x:x[1], reverse=True)[:int((1.0 - percentage)*len(probs_dictionary.items()))]
    else:
        pruned = sorted(probs_dictionary.items(), key = lambda x:x[1], reverse=True)[:topN]

    final_kept_candidates = {elem[0]: candidates[elem[0]] for elem in pruned if elem[1] > 0.5}

    #[elem[0] for elem in pruned if elem[1] > 0.5] # if elem[1] > 0.5, it was classified as 'keepable'
    time2 = time.time()
    print('time to predict what candidates to keep', (time2-time1)*1000.0, 'ms')
    print('initial number of candidates', len(candidates.keys()),
          'mid number of candidates', len(candidates_kept.keys()),
          'final number of candidates', len(final_kept_candidates))
    return final_kept_candidates

def check_efficiency_and_effectiveness(base_dataset,
                                       path_to_candidates,
                                       key, 
                                       target,
                                       training_data,
                                       rename_numerical=True,
                                       separator=SEPARATOR,
                                       feature_selector=recursive_feature_elimination, #rifs,
                                       prepruning=prune_candidates_hierarchical, 
                                       topN=100):
    '''
    This function gets the time to run a feature selector with and without
    pre-pruning using either the hierarchical classifier or classic prida
    '''
    
    print('Initial performance')
    compute_user_model_performance(base_dataset, target, base_dataset.drop([key, target], axis=1).columns)
    print('******* PRUNING WITH HIERARCHICAL CLASSIFIER ********')
    #Step 2: let's see how much time it takes to run the classifier-based pruner
    if prepruning == prune_candidates_hierarchical:
        candidates_to_keep = prune_candidates_hierarchical(training_data,  
                                              base_dataset,
                                              path_to_candidates,
                                              key, 
                                              target,
                                              topN=topN)
    elif prepruning == prune_candidates_classic:
        print('TODO')
    else:
        print('prepruner that was passed is not implemented')
        exit()
        
    augmented_dataset, names_and_columns = join_datasets(base_dataset.reset_index(), candidates_to_keep, key)
    print('candidates kept by hierarchical classifier', augmented_dataset.columns.to_list())

    #Step 3: select features with selector over pruned dataset (if RIFS, we inject 20% of random features)
    time1 = time.time()
    if feature_selector == rifs:
        selected_pruned = rifs(augmented_dataset,  
                               target, 
                               key) 
    elif feature_selector == recursive_feature_elimination:
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
        print('time to create and assess user\'s model with pruner', prepruning.__name__, (time2-time1)*1000.0, 'ms')

if __name__ == '__main__':    
    path_to_base_table = sys.argv[1]
    path_to_candidates = sys.argv[2]
    key = sys.argv[3]
    target = sys.argv[4]

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
                                       topN=100)
