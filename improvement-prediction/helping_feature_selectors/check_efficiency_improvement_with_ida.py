import pandas as pd
import json
import os
import warnings; warnings.simplefilter('ignore')
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.linear_model import LinearRegression

def join_datasets(base_dataset, dataset_directory, key, mean_data_imputation=True, rename_numerical=True, separator='|'):
    '''
    Given (1) a base dataset, (2) a directory with datasets that only have two 
    columns (one key and one numerical attribute), and (3) a key that is present 
    in all of them and helps for joining purposes, this function generates a big
    table composed of all joined datasets.
    '''
    
    augmented_dataset = base_dataset
    dataset_names = [f for f in os.listdir(dataset_directory) if '.csv' in f]
    for name in dataset_names:
        try:
            ### Step 1: read the dataset in the directory
            dataset = pd.read_csv(os.path.join(dataset_directory, name), 
                                  sep=separator)
            
            ### Step 2 (optional):  rename the numerical column in the dataset
            if rename_numerical:
                numerical_column = [i for i in dataset.columns if i != key][0]
                dataset = dataset.rename(columns={numerical_column: name.split('.')[0]})
    
            ### Step 3: augment the table
            #print('NAME', name)
            augmented_dataset = pd.merge(augmented_dataset, 
                                         dataset,
                                         how='left',
                                         on=key)
        except pd.errors.EmptyDataError:
            continue
    
    augmented_dataset = augmented_dataset.set_index(key)
    augmented_dataset = augmented_dataset.select_dtypes(include=['int64', 'float64'])
    if mean_data_imputation:
        fill_NaN = SimpleImputer(missing_values=np.nan, strategy='mean')
        new_data = pd.DataFrame(fill_NaN.fit_transform(augmented_dataset))
        new_data.index = augmented_dataset.index
        new_data.columns = augmented_dataset.columns
        return new_data
    
    return augmented_dataset

from sklearn.svm import SVC
from sklearn.metrics import r2_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler

FEATURES = ['query_num_of_columns', 'query_num_of_rows', 'query_row_column_ratio',
            'query_max_skewness', 'query_max_kurtosis', 'query_max_unique', 
            'candidate_num_rows', 'candidate_max_skewness', 'candidate_max_kurtosis',
            'candidate_max_unique', 'query_target_max_pearson', 
            'query_target_max_spearman', 'query_target_max_covariance', 
            'query_target_max_mutual_info', 'candidate_target_max_pearson', 
            'candidate_target_max_spearman', 'candidate_target_max_covariance', 
            'candidate_target_max_mutual_info']
THETA = 1

def train_rbf_svm(features, classes):
    '''
    Builds a model using features to predict associated classes,
    '''

    feature_scaler = MinMaxScaler().fit(features)
    features_train = feature_scaler.transform(features)
    clf = SVC(max_iter=1000, gamma='auto', probability=True)
    clf.fit(features_train, classes)

    return feature_scaler, clf

# The next two lines are important for importing files that are in the parent directory, 
# necessary to generate the features
import sys
sys.path.append('../')
from feature_factory import *


def compute_features(query_key_values,
                     candidate_dataset, 
                     key, 
                     target_name, 
                     augmented_dataset=pd.DataFrame([]),
                     mean_data_imputation=True):
    '''
    This function generates features required to determine, through classification, 
    whether an augmentation with the candidate_dataset (which is single-feature) is likely to 
    hamper the model (or simply bring no gain)
    '''
    
    ## In order, the returned features are number_of_columns, number_of_rows, row_to_column_ratio,
    ## max_mean, max_outlier_percentage, max_skewness, max_kurtosis, max_number_of_unique_values.
    ## For now, we're only using number_of_columns, number_of_rows, row_to_column_ratio, 
    ## max_skewness, max_kurtosis, max_number_of_unique_values, so we remove the unnecessary elements 
    ## in the lines below
 
    # Step 1: individual candidate features
    feature_factory_candidate = FeatureFactory(candidate_dataset)
    candidate_dataset_individual_features = feature_factory_candidate.get_individual_features(func=max_in_modulus)
    ## For now, we're only using number_of_rows, max_skewness, max_kurtosis, max_number_of_unique_values, 
    ## so we remove the unnecessary elements in the lines below 
    candidate_dataset_individual_features = [candidate_dataset_individual_features[index] for index in [1, 5, 6, 7]]

    # Step 2: join the datasets and compute pairwise features
    if augmented_dataset.empty:
        augmented_dataset = pd.merge(query_dataset, 
                                     candidate_dataset,
                                     how='left',
                                     on=key)
    #augmented_dataset = augmented_dataset.set_index(key)
    if mean_data_imputation:
        fill_NaN = SimpleImputer(missing_values=np.nan, strategy='mean')
        new_dataset = pd.DataFrame(fill_NaN.fit_transform(augmented_dataset))
        new_dataset.columns = augmented_dataset.columns
        new_dataset.index = augmented_dataset.index
        augmented_dataset = new_dataset
    
    # Step 3: get candidate-target features
    ## The features are, in order: max_query_candidate_pearson, max_query_candidate_spearman, 
    ## max_query_candidate_covariance, max_query_candidate_mutual_info
    column_names = candidate_dataset.columns.tolist() + [target_name]
    feature_factory_candidate_target = FeatureFactory(augmented_dataset[column_names])
    candidate_features_target = feature_factory_candidate_target.get_pairwise_features_with_target(target_name,
                                                                                                   func=max_in_modulus)
     # Step 4: get query-candidate feature "containment ratio". We may not use it in models, but it's 
    ## important to have this value in order to filter candidates in baselines, for example.
    candidate_key_values = candidate_dataset.index.values
    intersection_size = len(set(query_key_values) & set(candidate_key_values))
    containment_ratio = [intersection_size/len(query_key_values)]

    return candidate_dataset_individual_features, candidate_features_target, containment_ratio

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

def  compute_adjusted_r2_score(number_of_regressors, r2_score, number_of_samples):
    '''
    This function adjusts the value of an r2 score based on its numbers 
    of regressors and samples.
    '''
    try:
        num = (1.0 - r2_score) * (number_of_samples - 1.0)
        den = number_of_samples - number_of_regressors - 1.0
        return 1.0 - (num/den)  
    except ZeroDivisionError:
        print('**** ERROR', number_of_samples, number_of_regressors)
        return r2_score
    
def compute_model_performance_improvement(query_dataset, 
                                          candidate_dataset, 
                                          target_name, 
                                          key, 
                                          mean_data_imputation=True, 
                                          adjusted_r2_score=False, 
                                          model=None):
    '''
    This function computes the change in (adjusted) R2 score obtained when we try to predict 'target_name' with an 
    augmented dataset (query_dataset + candidate_dataset)
    '''
    
    # To make sure that we compare apples to apples, let's perform the augmentation and any 
    # necessary missing data imputation first
    augmented_dataset = pd.merge(query_dataset, 
                                 candidate_dataset,
                                 how='left',
                                 on=key)
    
    if mean_data_imputation:
        fill_NaN = SimpleImputer(missing_values=np.nan, strategy='mean')
        new_dataset = pd.DataFrame(fill_NaN.fit_transform(augmented_dataset))
        new_dataset.columns = augmented_dataset.columns
        new_dataset.index = augmented_dataset.index
        augmented_dataset = new_dataset
    
    # Now let's split the data
    X_train, X_test, y_train, y_test = train_test_split(augmented_dataset.drop([target_name], axis=1), 
                                                        augmented_dataset[target_name], 
                                                        test_size=0.33, 
                                                        random_state=42)
    
    # Computing the initial and final r-squared scores
    if not model:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    model.fit(X_train[query_dataset.drop([target_name], axis=1).columns], y_train.ravel())
    y_pred_initial = model.predict(X_test[query_dataset.drop([target_name], axis=1).columns])
    r2_score_initial = r2_score(y_test, y_pred_initial)
    if adjusted_r2_score:
        r2_score_initial = compute_adjusted_r2_score(len(query_dataset.drop([target_name], axis=1).columns),
                                                     r2_score_initial, 
                                                     len(y_test))
        
    model.fit(X_train[augmented_dataset.drop([target_name], axis=1).columns], y_train.ravel())
    y_pred_final = model.predict(X_test[augmented_dataset.drop([target_name], axis=1).columns])
    r2_score_final = r2_score(y_test, y_pred_final)
    if adjusted_r2_score:
        r2_score_final = compute_adjusted_r2_score(len(augmented_dataset.drop([target_name], axis=1).columns),
                                                   r2_score_final,
                                                   len(y_test))
    
    performance_difference = (r2_score_final - r2_score_initial)/np.fabs(r2_score_initial)
    
    return r2_score_initial, r2_score_final, performance_difference


def normalize_features(features, scaler=None):
    '''
    This function normalizes features using sklearn's StandardScaler
    '''
    if not scaler:
        scaler = MinMaxScaler().fit(features)
    return scaler.transform(features)


from sparsereg.model.base import STRidge # the stype of sparsereg they use is not clear in the ARDA paper
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split

RANDOM_PREFIX = 'random_feature_'
MAX_ROWS_FOR_ESTIMATION = 5000

def augment_with_random_features(dataset, target_name, number_of_random_features):
    '''
    Given a dataset, the name of the target, and a number of random features, this function derives
    random features that are based on the original ones in the dataset
    '''
    #print('in augment_with_random_features')
    if dataset.shape[0] > MAX_ROWS_FOR_ESTIMATION:
        dataset = dataset.sample(n=MAX_ROWS_FOR_ESTIMATION, random_state=42)
    mean = dataset.drop([target_name], axis=1).T.mean()
    #print('MEAN SHAPE', mean.shape)
    cov = dataset.drop([target_name], axis=1).T.cov()
    #print('COVARIANCE', cov)
    features = np.random.multivariate_normal(mean, cov, number_of_random_features)
    for i in range(number_of_random_features):
        dataset[RANDOM_PREFIX + str(i)] = features[i,:]
    #print('leaving augment_with_random_features')
    return dataset

def combine_rankings(rf_coefs, regression_coefs, feature_names, lin_comb_coef=0.5):
    '''
    Given feature coefficients computed with different methods (random forest and a regularized 
    regression), we scale their values, combine them linearly, rescale them and rank their  
    corresponding features according to their values
    '''
    
    # random forest coefficients are already in interval [0, 1]; let's pre-scale the regression 
    # coefficients, which are all over the place
    min_value = min(regression_coefs)
    shifted = np.array([elem + min_value*-1 for elem in regression_coefs])
    normalized_regression_coefs = shifted/sum(shifted)
    
    combined_scores = lin_comb_coef*rf_coefs + (1-lin_comb_coef)*normalized_regression_coefs
    ranked_features = sorted([(feat_name, score) for feat_name, score in zip(feature_names, combined_scores)], 
                             key=lambda x: x[1], 
                             reverse=True)
    return [elem[0] for elem in ranked_features]

def feature_in_front_of_random(feature_name, ranking):
    '''
    This function checks whether there are random features before 
    feature_name
    '''
    random_not_seen = True
    for feat in ranking:
        if RANDOM_PREFIX in feat:
            return 0
        if feat == feature_name:
            return 1
    return 0

def aggregate_features_by_quality(rankings):
    '''
    Given feature rankings, this function counts, for every non-random feature, 
    how many times it occured before every random feature
    '''
    feature_names = [elem for elem in rankings[0] if RANDOM_PREFIX not in elem]
    
    feats_quality = {}
    for feature in feature_names:
        for rank in rankings:
            if feature in feats_quality:
                feats_quality[feature] += feature_in_front_of_random(feature, rank)
            else:
                feats_quality[feature] = feature_in_front_of_random(feature, rank)
    sorted_feats =  sorted(feats_quality.items(), 
                           key=lambda x: x[1], 
                           reverse=True)
    return [(elem[0], elem[1]/len(rankings)) for elem in sorted_feats]
    
def random_injection_feature_selection(augmented_dataset,
                                       target_name,  
                                       tau, 
                                       eta, 
                                       k_random_seeds):
    '''
    This is ARDA's feature selection algorithm RIFS. Given an augmented dataset, ideally 
    created through joins over sketches over the original datasets, a threshold tau for 
    quality of features, a fraction eta of random features to inject, and k random seeds to perform 
    k experiments in a reproducible way, it selects the features that should be used in the augmented 
    dataset
    '''
    #print('in random injection')
    number_of_random_features = int(np.ceil(eta*augmented_dataset.drop([target_name], axis=1).shape[1]))
    #print('number of random features', number_of_random_features)
    augmented_dataset_with_random = augment_with_random_features(augmented_dataset,
                                                                 target_name, 
                                                                 number_of_random_features)
    
    # Now we obtain rankings using random forests and sparse regression models
    ## the paper does not say what hyperparameters were used in the experiment
    rankings = []
    for seed in k_random_seeds:
        #print('seed', seed)
        rf = RandomForestRegressor(n_estimators=100, random_state=seed)
        rf.fit(augmented_dataset_with_random.drop([target_name], axis=1), augmented_dataset_with_random[target_name])
        rf_coefs = rf.feature_importances_
        #print('done fitting random forest')
        ## coef = STRidge().fit(augmented_dataset_features, target_column_data).coef_
        ## print(coef, augmented_dataset_features.columns)
        ## print(len(rf.feature_importances_))
        
        ## This version of lasso is giving lower weights to random features, which is good
        lasso = Lasso(random_state=seed)
        lasso.fit(augmented_dataset_with_random.drop([target_name], axis=1), augmented_dataset_with_random[target_name])
        lasso_coefs = lasso.coef_
        #print('done fitting lasso')
        rank = combine_rankings(rf_coefs, lasso_coefs, augmented_dataset_with_random.drop([target_name], axis=1).columns)
        rankings.append(rank)
    
    # Now, for each non-random feature, we get the number of times it appeared in front of 
    ## all random features
    sorted_features = aggregate_features_by_quality(rankings)
    return [elem[0] for elem in sorted_features if elem[1] >= tau]
    
def wrapper_algorithm(augmented_dataset, target_name, key, thresholds_T, eta, k_random_seeds):
    '''
    This function searches for the best subset of features by doing an exponential search
    '''
    X_train, X_test, y_train, y_test = train_test_split(augmented_dataset.drop(target_name, axis=1), 
                                                        augmented_dataset[target_name], 
                                                        test_size=0.33,
                                                        random_state=42)
    current_r2_score = float('-inf')
    linreg = LinearRegression()
    selected = []
    #print('wrapper algorithm')
    for tau in thresholds_T:
        #print('tau', tau)
        selected = random_injection_feature_selection(augmented_dataset,
                                                      target_name, 
                                                      tau, 
                                                      eta, 
                                                      k_random_seeds)
        #print('got out of random injection')
        linreg.fit(X_train[selected], y_train)
        y_pred = linreg.predict(X_test[selected])
        new_r2_score = r2_score(y_test, y_pred)
        if  new_r2_score > current_r2_score:
            current_r2_score = new_r2_score
        else:
            break
    return selected


import time

def check_efficiency_with_ida(base_dataset, 
                              dataset_directory, 
                              key, 
                              target_name, 
                              training_data, 
                              thresholds_tau=[0.2, 0.4, 0.6, 0.8], #REFACTOR: this parameter is only used by RIFS
                              eta=0.2, #REFACTOR: this parameter is only used by RIFS
                              k_random_seeds=[42, 17, 23, 2, 5, 19, 37, 41, 13, 33], #REFACTOR: this parameter is only used by RIFS
                              mean_data_imputation=True, 
                              rename_numerical=True, 
                              separator='|', 
                              feature_selector=wrapper_algorithm,
                              gain_prob_threshold=0.5):
    '''
    This function compares the time to run a feature selector with and without pre-pruning with IDA
    '''
    
    #Step 1: do the join with every candidate dataset in dataset_directory. 
    ## This has to be done both with and without IDA. 
    augmented_dataset = join_datasets(base_dataset, dataset_directory, key, rename_numerical=rename_numerical, separator=separator)
    augmented_dataset = augmented_dataset.loc[:,~augmented_dataset.columns.duplicated()] #removing duplicate columns
    print('Done creating the augmented dataset')
    
    #Step 2: let's see how much time it takes to select features with RIFS, injecting 20% of random features
    time1 = time.time()
    if feature_selector == wrapper_algorithm:
        selected_all = wrapper_algorithm(augmented_dataset, target_name, key, thresholds_tau, eta, k_random_seeds)
    elif feature_selector == boruta_algorithm:
        selected_all = boruta_algorithm(augmented_dataset, target_name)
    elif feature_selector == stepwise_selection:
        selected_all = stepwise_selection(augmented_dataset.drop([target_name], axis=1), augmented_dataset[target_name])
    elif feature_selector == recursive_feature_elimination:
        selected_all = recursive_feature_elimination(augmented_dataset.drop([target_name], axis=1), augmented_dataset[target_name])
    else:
        print('feature selector that was passed is not implemented')
        exit()
    time2 = time.time()
    print('time to run feature selector', (time2-time1)*1000.0, 'ms')
    
    #Step 3: let's train our IDA model over the training dataset
    time1 = time.time()
    feature_scaler, model = train_rbf_svm(training_data[FEATURES], 
                                          training_data['class_pos_neg'])
    time2 = time.time()
    print('time to train our model', (time2-time1)*1000.0, 'ms')
    
    #Step 4: generate a label for every feature in the augmented dataset
    time1 = time.time()
    candidate_names = set(augmented_dataset.columns) - set(base_dataset.columns)
    feature_vectors = []

    ## We just need to compute query features once!
    ## In order, the returned features are number_of_columns, number_of_rows, row_to_column_ratio,
    ## max_mean, max_outlier_percentage, max_skewness, max_kurtosis, max_number_of_unique_values.
    ## For now, we're only using number_of_columns, number_of_rows, row_to_column_ratio, 
    ## max_skewness, max_kurtosis, max_number_of_unique_values, so we remove the unnecessary elements 
    ## in the lines below
    feature_factory_query = FeatureFactory(base_dataset.set_index(key).drop([target_name], axis=1))
    query_features = feature_factory_query.get_individual_features(func=max_in_modulus)
    query_features = [query_features[index] for index in [0, 1, 2, 5, 6, 7]]

    ## get query-target features 
    ## The features are, in order: max_query_target_pearson, max_query_target_spearman, 
    ## max_query_target_covariance, max_query_target_mutual_info
    feature_factory_full_query = FeatureFactory(base_dataset.set_index(key))
    query_features_target = feature_factory_full_query.get_pairwise_features_with_target(target_name, 
                                                                                         func=max_in_modulus)
    query_key_values = base_dataset.set_index(key).index.values
        
    for name in candidate_names:
        candidate_dataset = augmented_dataset.reset_index()[[key, name]]
        #print('getting features for candidate column', name)
        #print(candidate_dataset.set_index(key))
        candidate_features, candidate_features_target, containment_ratio = compute_features(query_key_values,
                                                                                            candidate_dataset.set_index(key), 
                                                                                            key, 
                                                                                            target_name, 
                                                                                            augmented_dataset=augmented_dataset)
        feature_vectors.append(query_features + candidate_features + query_features_target + candidate_features_target)
    predictions = model.predict(normalize_features(np.array(feature_vectors))) 
    gain_pred_probas = [elem[0] for elem in model.predict_proba(normalize_features(np.array(feature_vectors)))]
    probs_dictionary = {name: prob for name, prob in zip(candidate_names, list(gain_pred_probas))}
    
    candidates_to_keep = [name for name, pred, prob in zip(candidate_names, predictions, gain_pred_probas) if pred == 'gain' and prob >= gain_prob_threshold]
    time2 = time.time()
    print('time to predict what candidates to keep', (time2-time1)*1000.0, 'ms')
    
    #Step 5: run RIFS only considering the features to keep
    #print('initial columns', base_dataset.set_index(key).columns.to_list())
    #print('candidates to keep', candidates_to_keep)
    #print('augmented columns', augmented_dataset.columns)
    pruned = augmented_dataset[base_dataset.set_index(key).columns.to_list() + candidates_to_keep]
    
    time1 = time.time()
    if feature_selector == wrapper_algorithm:
        selected_pruned = wrapper_algorithm(pruned, 
                                            target_name, 
                                            key, 
                                            thresholds_tau, 
                                            eta, 
                                            k_random_seeds)
    elif feature_selector == boruta_algorithm:
        selected_pruned = boruta_algorithm(pruned, target_name)
    elif feature_selector == stepwise_selection:
        selected_pruned = stepwise_selection(pruned.drop([target_name], axis=1), pruned[target_name])
    elif feature_selector == recursive_feature_elimination:
        selected_pruned = recursive_feature_elimination(pruned.drop([target_name], axis=1), pruned[target_name])
    else:
        print('feature selector that was passed is not implemented')
        exit()  
    time2 = time.time()
    print('time to run feature selector over features to keep', (time2-time1)*1000.0, 'ms')
    print('size of entire dataset', augmented_dataset.shape[1], 'size of pruned', pruned.shape[1])
    print('size of selected features when you use the entire dataset', len(selected_all))
    print('size of selected features when you use the pruned dataset', len(selected_pruned))
    print('size of candidates that were selected to be kept', len(candidates_to_keep))
    print('total number of candidates', len(candidate_names))
    return selected_all, candidates_to_keep, selected_pruned, model, probs_dictionary

from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier
def boruta_algorithm(dataset, target_name):
    '''
    This function selects features in the dataset using an implementation 
    of the boruta algorithm
    '''
    print('USING BORUTA')
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    feat_selector = BorutaPy(rf, n_estimators='auto', random_state=1)
    feat_selector.fit(dataset.drop([target_name], axis=1).values, dataset[target_name].values.ravel())
    filtered = feat_selector.transform(dataset.drop([target_name], axis=1).values)
    generously_selected = feat_selector.support_weak_
    feat_names = dataset.drop([target_name], axis=1).columns
    return [name for name, mask in zip(feat_names, generously_selected) if mask]

def compute_model_performance(dataset, features, target_name):
    '''
    This function checks how well a random forest, trained on a given set of features, 
    performs in the prediction of a target
    '''

    # Now let's split the data
    X_train, X_test, y_train, y_test = train_test_split(dataset.drop([target_name], axis=1),
                                                        dataset[target_name],
                                                        test_size=0.33,
                                                        random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train[features], y_train.ravel())
    y_pred = model.predict(X_test[features])
    print(r2_score(y_test, y_pred))

from mlxtend.feature_selection import SequentialFeatureSelector as sfs
def stepwise_selection(data, target):
    print('USING STEPWISE_SELECTION')
    # Build RF regressor  to use in feature selection
    rf = RandomForestRegressor(n_estimators=10, n_jobs=-1, random_state=42)
    # Build step forward feature selection
    sfs1 = sfs(rf,
               k_features=5,
               forward=True,
               floating=True,
               scoring='r2',
               cv=5)

    # Perform SFFS
    sfs1 = sfs1.fit(data, target)
    all_features = data.columns.tolist()
    chosen_features = list(sfs1.k_feature_idx_)
    return [all_features[i] for i in chosen_features]

from sklearn.feature_selection import RFE
def recursive_feature_elimination(data, target):
    estimator = LinearRegression()
    selector = RFE(estimator)
    reduced = selector.fit(data, target)
    return [elem for elem, label in zip(list(data.columns), list(reduced.support_)) if label]

def assess_classifier_quality(classifier, 
                              base_dataset, 
                              dataset_directory, 
                              key, 
                              target_name, 
                              rename_numerical=True, 
                              separator='|'):
    '''
    This function generates true labels and predictions for a set of candidate datasets and 
    assesses the quality of the classifier
    '''
    
    augmented_dataset = join_datasets(base_dataset, 
                                      dataset_directory, 
                                      key, 
                                      rename_numerical=rename_numerical, 
                                      separator=separator)
    augmented_dataset = augmented_dataset.loc[:,~augmented_dataset.columns.duplicated()]
    
    candidate_names = set(augmented_dataset.columns) - set(base_dataset.columns)
    feature_vectors = []
    labels = []
    improvements = {}
    feature_factory_query = FeatureFactory(base_dataset.set_index(key).drop([target_name], axis=1))
    query_features = feature_factory_query.get_individual_features(func=max_in_modulus)
    query_features = [query_features[index] for index in [0, 1, 2, 5, 6, 7]]
    feature_factory_full_query = FeatureFactory(base_dataset.set_index(key))
    query_features_target = feature_factory_full_query.get_pairwise_features_with_target(target_name, 
                                                                                         func=max_in_modulus)
    query_key_values = base_dataset.set_index(key).index.values
    for name in candidate_names:
        candidate_dataset = augmented_dataset.reset_index()[[key, name]]
        candidate_features, candidate_features_target, containment_ratio = compute_features(query_key_values,
                                                                                            candidate_dataset.set_index(key), 
                                                                                            key, 
                                                                                            target_name, 
                                                                                            augmented_dataset=augmented_dataset)
        feature_vectors.append(query_features + candidate_features + query_features_target + candidate_features_target)
        
        initial, final, improvement = compute_model_performance_improvement(base_dataset.set_index(key), 
                                                                            candidate_dataset.set_index(key), 
                                                                            target_name, 
                                                                            key, 
                                                                            adjusted_r2_score=False)
        if improvement > 0: 
            labels.append('gain')
        else:
            labels.append('loss')
        improvements[name] = improvement
    predictions = classifier.predict(normalize_features(np.array(feature_vectors))) 
    #print(labels)
    #print(predictions)
    print(classification_report(labels, predictions))
    return improvements

import matplotlib.pyplot as plt
NUM_BINS = 50
def plot_histogram(data, xlabel, ylabel, title, figname):
    '''
    Simple histogram-plotting function
    '''
    plt.hist(data, bins=NUM_BINS, alpha=0.7, color='blue')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(figname, dpi=600)


if __name__ == '__main__':
    openml_training = pd.read_csv('../classification/training-simplified-data-generation.csv')
    openml_training['class_pos_neg'] = ['gain' if row['gain_in_r2_score'] > 0 else 'loss'
                                        for index, row in openml_training.iterrows()]
    openml_training_high_containment = openml_training.loc[openml_training['containment_fraction'] >= THETA]

    feature_scaler, model = train_rbf_svm(openml_training_high_containment[FEATURES], 
                                          openml_training_high_containment['class_pos_neg'])

    flight_query_dataset = pd.read_csv('arda_datasets/airline/flights.csv')
    flight_query_dataset = flight_query_dataset.set_index('key').select_dtypes(include=['int64', 'float64'])


    # selected_all_airplane, candidates_to_keep_airplane, selected_pruned_airplane, model, gain_probs = check_efficiency_with_ida(flight_query_dataset.reset_index(), 
    #                                                                                                                 'arda_datasets/airline/candidates/', 
    #                                                                                                                 'key', 
    #                                                                                                                 'population', 
    #                                                                                                                 openml_training_high_containment, 
    #                                                                                                                 rename_numerical=False, 
    #                                                                                                                 separator=',')
    # plot_histogram(gain_probs.values(), 'Probability of Gain', 'Percentage', 'Probabilities of Gain -- Pickup Use Case', 'pickup_gain_probs.png')
    # print('FEATURES SELECTED FROM ENTIRE DATASET ALONG WITH PROBABILITIES -- PICKUP', [(candidate, gain_probs[candidate]) for candidate in selected_all_airplane if candidate in gain_probs])
    # print('FEATURES SELECTED FROM PRUNED DATASET ALONG WITH PROBABILITIES -- PICKUP', [(candidate, gain_probs[candidate]) for candidate in selected_pruned_airplane  if candidate in gain_probs])    
    #print('FEATURES SELECTED FROM ENTIRE DATASET -- AIRPLANE', selected_all_airplane)
    #print('FEATURES SELECTED FROM THE PRUNED DATASET -- AIRPLANE', selected_pruned_airplane)
    ## THE LINE BELOW TAKES A LONG TIME TO RUN!
    # improvements = assess_classifier_quality(model,
    #                           flight_query_dataset.reset_index(),
    #                           'arda_datasets/airline/candidates/',
    #                           'key',
    #                           'population',
    #                           rename_numerical=False,
    #                           separator=',')
    # plot_histogram(improvements.values(), 'Performance Improvement', 'Percentage', 'Performance Improvements -- Pickup Use Case', 'pickup_gain_improvements.png')
    # print('FEATURES SELECTED FROM ENTIRE DATASET ALONG WITH IMPROVEMENTS -- PICKUP', [(candidate, improvements[candidate]) for candidate in selected_all_airplane  if candidate in improvements])
    # print('FEATURES SELECTED FROM PRUNED DATASET ALONG WITH IMPROVEMENTS -- PICKUP', [(candidate, improvements[candidate]) for candidate in selected_pruned_airplane  if candidate in improvements])

    
    initial_college_dataset = pd.read_csv('datasets_for_use_cases/companion-datasets/college-debt-v2.csv')
    initial_college_dataset = initial_college_dataset.fillna(initial_college_dataset.mean())
    # selected_all, candidates_to_keep, selected_pruned, model, gain_probs = check_efficiency_with_ida(initial_college_dataset, 
    #                                                                                      'datasets_for_use_cases/companion-datasets/college-debt-single-column/', 
    #                                                                                      'UNITID', 
    #                                                                                      'DEBT_EARNINGS_RATIO', 
    #                                                                                      openml_training_high_containment, 
    #                                                                                      rename_numerical=False, 
    #                                                                                      separator=',')
    # plot_histogram(gain_probs.values(), 'Probability of Gain', 'Percentage', 'Probabilities of Gain -- College Use Case', 'college_gain_probs.png')
    # print('FEATURES SELECTED FROM ENTIRE DATASET ALONG WITH PROBABILITIES -- COLLEGE', [(candidate, gain_probs[candidate]) for candidate in selected_all if candidate in gain_probs])
    # print('FEATURES SELECTED FROM PRUNED DATASET ALONG WITH PROBABILITIES -- COLLEGE', [(candidate, gain_probs[candidate]) for candidate in selected_pruned  if candidate in gain_probs])    
    # #print('FEATURES SELECTED FROM ENTIRE DATASET -- COLLEGE', selected_all)
    # #print('FEATURES SELECTED FROM THE PRUNED DATASET -- COLLEGE', selected_pruned)
    # improvements = assess_classifier_quality(model,
    #                           initial_college_dataset,
    #                           'datasets_for_use_cases/companion-datasets/college-debt-single-column/', 
    #                           'UNITID', 
    #                           'DEBT_EARNINGS_RATIO',
    #                           rename_numerical=False,
    #                           separator=',')
    
    # plot_histogram(improvements.values(), 'Performance Improvement', 'Percentage', 'Performance Improvements -- College Use Case', 'college_gain_improvements.png')
    # print('FEATURES SELECTED FROM ENTIRE DATASET ALONG WITH IMPROVEMENTS -- COLLEGE', [(candidate, improvements[candidate]) for candidate in selected_all  if candidate in improvements])
    # print('FEATURES SELECTED FROM PRUNED DATASET ALONG WITH IMPROVEMENTS -- COLLEGE', [(candidate, improvements[candidate]) for candidate in selected_pruned  if candidate in improvements])
    
    crash_many_predictors = pd.read_csv('crash_many_predictors.csv', sep=SEPARATOR)
    # selected_all_crash, candidates_to_keep_crash, selected_pruned_crash, model, gain_probs = check_efficiency_with_ida(crash_many_predictors,
    #                                                                                                                       'nyc_indicators/',
    #                                                                                                                       'time',
    #                                                                                                                       'crash_count',
    #                                                                                                                       openml_training_high_containment)
    # plot_histogram(gain_probs.values(), 'Probability of Gain', 'Percentage', 'Probabilities of Gain -- Crash Use Case', 'crash_gain_probs.png')
    # print('FEATURES SELECTED FROM ENTIRE DATASET ALONG WITH PROBABILITIES -- CRASH', [(candidate, gain_probs[candidate]) for candidate in selected_all_crash if candidate in gain_probs])
    # print('FEATURES SELECTED FROM PRUNED DATASET ALONG WITH PROBABILITIES -- CRASH', [(candidate, gain_probs[candidate]) for candidate in selected_pruned_crash  if candidate in gain_probs])    
    # # print('FEATURES SELECTED FROM ENTIRE DATASET -- CRASH', selected_all_crash)
    # # print('FEATURES SELECTED FROM THE PRUNED DATASET -- CRASH', selected_pruned_crash)
    # improvements = assess_classifier_quality(model,
    #                                          crash_many_predictors,
    #                                          'nyc_indicators/',
    #                                          'time',
    #                                          'crash_count')
    # plot_histogram(improvements.values(), 'Performance Improvement', 'Percentage', 'Performance Improvements -- Crash Use Case', 'crash_gain_improvements.png')
    # print('FEATURES SELECTED FROM ENTIRE DATASET ALONG WITH IMPROVEMENTS -- CRASH', [(candidate, improvements[candidate]) for candidate in selected_all_crash  if candidate in improvements])
    # print('FEATURES SELECTED FROM PRUNED DATASET ALONG WITH IMPROVEMENTS -- CRASH', [(candidate, improvements[candidate]) for candidate in selected_pruned_crash  if candidate in improvements])    

    
    # selected_all_airplane_boruta, candidates_to_keep_airplane_boruta, selected_pruned_airplane_boruta, model, gain_probs = check_efficiency_with_ida(flight_query_dataset.reset_index(), 
    #                                                                                                                                      'arda_datasets/airline/candidates/', 
    #                                                                                                                                      'key', 
    #                                                                                                                                      'population', 
    #                                                                                                                                      openml_training_high_containment, 
    #                                                                                                                                      rename_numerical=False, 
    #                                                                                                                                      separator=',',
    #                                                                                                                                      feature_selector=boruta_algorithm)
    # print('FEATURES SELECTED FROM ENTIRE DATASET -- AIRPLANE -- BORUTA', selected_all_airplane_boruta)
    # print('FEATURES SELECTED FROM THE PRUNED DATASET -- AIRPLANE -- BORUTA', selected_pruned_airplane_boruta)
    
    # selected_all_college_boruta, candidates_to_keep_college_boruta, selected_pruned_college_boruta, model, gain_probs = check_efficiency_with_ida(initial_college_dataset,
    #                                                                                                                                   'datasets_for_use_cases/companion-datasets/college-debt-single-column/',
    #                                                                                                                                   'UNITID',
    #                                                                                                                                   'DEBT_EARNINGS_RATIO',
    #                                                                                                                                   openml_training_high_containment,
    #                                                                                                                                   rename_numerical=False,
    #                                                                                                                                   separator=',',
    #                                                                                                                                   feature_selector=boruta_algorithm)
    
    # print('FEATURES SELECTED FROM ENTIRE DATASET -- COLLEGE -- BORUTA', selected_all_college_boruta)
    # print('FEATURES SELECTED FROM THE PRUNED DATASET -- COLLEGE -- BORUTA', selected_pruned_college_boruta)

    # selected_all_boruta, candidates_to_keep_boruta, selected_pruned_boruta, model, gain_probs = check_efficiency_with_ida(crash_many_predictors, 
    #                                                                                                           'nyc_indicators/', 
    #                                                                                                           'time', 
    #                                                                                                           'crash_count', 
    #                                                                                                           openml_training_high_containment, 
    #                                                                                                           feature_selector=boruta_algorithm)
    # print('FEATURES SELECTED FROM ENTIRE DATASET -- CRASH -- BORUTA', selected_all_boruta)
    # print('FEATURES SELECTED FROM THE PRUNED DATASET -- CRASH -- BORUTA', selected_pruned_boruta)


    # selected_all_airplane_rfe, candidates_to_keep_airplane_rfe, selected_pruned_airplane_rfe, model, gain_probs = check_efficiency_with_ida(flight_query_dataset.reset_index(),    #                                                                                                                             'arda_datasets/airline/candidates/', 
    #                                                                                                                             'key', 
    #                                                                                                                             'population', 
    #                                                                                                                             openml_training_high_containment, 
    #                                                                                                                             rename_numerical=False, 
    #                                                                                                                             separator=',',
    #                                                                                                                             feature_selector=recursive_feature_elimination)
    # print('FEATURES SELECTED FROM ENTIRE DATASET -- AIRPLANE -- RFE', selected_all_airplane_rfe)
    # print('FEATURES SELECTED FROM THE PRUNED DATASET -- AIRPLANE -- RFE', selected_pruned_airplane_rfe)
    
    # selected_all_college_rfe, candidates_to_keep_college_rfe, selected_pruned_college_rfe, model, gain_probs = check_efficiency_with_ida(initial_college_dataset,
    #                                                                                                                          'datasets_for_use_cases/companion-datasets/college-debt-single-column/',
    #                                                                                                                          'UNITID',
    #                                                                                                                          'DEBT_EARNINGS_RATIO',
    #                                                                                                                          openml_training_high_containment,
    #                                                                                                                          rename_numerical=False,
    #                                                                                                                          separator=',',
    #                                                                                                                          feature_selector=recursive_feature_elimination)
    
    # print('FEATURES SELECTED FROM ENTIRE DATASET -- COLLEGE -- RFE', selected_all_college_rfe)
    # print('FEATURES SELECTED FROM THE PRUNED DATASET -- COLLEGE -- RFE', selected_pruned_college_rfe)

    # selected_all_rfe, candidates_to_keep_rfe, selected_pruned_rfe, model, gain_probs = check_efficiency_with_ida(crash_many_predictors, 
    #                                                                                                  'nyc_indicators/', 
    #                                                                                                  'time', 
    #                                                                                                  'crash_count', 
    #                                                                                                  openml_training_high_containment, 
    #                                                                                                  feature_selector=recursive_feature_elimination)
    # print('FEATURES SELECTED FROM ENTIRE DATASET -- CRASH -- RFE', selected_all_rfe)
    # print('FEATURES SELECTED FROM THE PRUNED DATASET -- CRASH -- RFE', selected_pruned_rfe)


    
    selected_all_airplane_stepwise, candidates_to_keep_airplane_stepwise, selected_pruned_airplane_stepwise, model = check_efficiency_with_ida(flight_query_dataset.reset_index(), 
                                                                                                                                'arda_datasets/airline/candidates/', 
                                                                                                                                'key', 
                                                                                                                                'population', 
                                                                                                                                openml_training_high_containment, 
                                                                                                                                rename_numerical=False, 
                                                                                                                                separator=',',
                                                                                                                                               feature_selector=stepwise_selection)
    print('FEATURES SELECTED FROM ENTIRE DATASET -- AIRPLANE -- STEPWISE', selected_all_airplane_stepwise)
    print('FEATURES SELECTED FROM THE PRUNED DATASET -- AIRPLANE -- STEPWISE', selected_pruned_airplane_stepwise)
    
    selected_all_college_stepwise, candidates_to_keep_college_stepwise, selected_pruned_college_stepwise, model = check_efficiency_with_ida(initial_college_dataset,
                                                                                                                             'datasets_for_use_cases/companion-datasets/college-debt-single-column/',
                                                                                                                             'UNITID',
                                                                                                                             'DEBT_EARNINGS_RATIO',
                                                                                                                             openml_training_high_containment,
                                                                                                                             rename_numerical=False,
                                                                                                                             separator=',',
                                                                                                                                            feature_selector=stepwise_selection)
    
    print('FEATURES SELECTED FROM ENTIRE DATASET -- COLLEGE -- STEPWISE', selected_all_college_stepwise)
    print('FEATURES SELECTED FROM THE PRUNED DATASET -- COLLEGE -- STEPWISE', selected_pruned_college_stepwise)

    selected_all_stepwise, candidates_to_keep_stepwise, selected_pruned_stepwise, model = check_efficiency_with_ida(crash_many_predictors, 
                                                                                                     'nyc_indicators/', 
                                                                                                     'time', 
                                                                                                     'crash_count', 
                                                                                                     openml_training_high_containment, 
                                                                                                                    feature_selector=stepwise_selection)
    print('FEATURES SELECTED FROM ENTIRE DATASET -- CRASH -- STEPWISE', selected_all_stepwise)
    print('FEATURES SELECTED FROM THE PRUNED DATASET -- CRASH -- STEPWISE', selected_pruned_stepwise)
