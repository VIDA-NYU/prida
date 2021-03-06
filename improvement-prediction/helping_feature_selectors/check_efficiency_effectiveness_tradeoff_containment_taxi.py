import time
import pandas as pd
import json
import os
import warnings; warnings.simplefilter('ignore')
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.linear_model import LinearRegression

def join_datasets(base_dataset, 
                  dataset_directory, 
                  base_key, 
                  mean_data_imputation=True, 
                  rename_numerical=True, 
                  separator='|', 
                  prepruning=None, 
                  percentage=0.5, 
                  candidate_key_columns=None,
                  topN=None):
    '''
    Given (1) a base dataset, (2) a directory with datasets that only have two 
    columns (one key and one numerical attribute), and (3) keys  for joining purposes, 
    this function generates a big table composed of all joined datasets.

    If prepruning == 'containment', the augmentation only happens including the 
    top 'percentage' overlapping candidates.
    '''

    print('inside join datasets. Prepruning is', prepruning)
    time1 = time.time()
    augmented_dataset = base_dataset
    #print(augmented_dataset.columns)
    #augmented_dataset.set_index(base_key, inplace=True)
    dataset_names = [f for f in os.listdir(dataset_directory)]
    
    containments = {}
    for name in dataset_names:
        #print('here is candidate', name)
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
    
            ### Step 3: augment the table
            if prepruning == 'containment':
                base_keys = set(base_dataset.index.values) #[base_key])
                if candidate_key_columns:
                    candidate_keys = set(dataset[candidate_key_columns[name]])
                else:
                    candidate_keys = set(dataset[base_key])
                intersection_size = len(base_keys & candidate_keys)
                containment_ratio = intersection_size/len(base_keys)
                containments[name] = containment_ratio
            else:
                if candidate_key_columns:
                    #print(name, dataset.columns, candidate_key_columns[name])
                    augmented_dataset = pd.merge(augmented_dataset,
                                                 dataset,
                                                 how='left',
                                                 left_on=[base_key],
                                                 right_on=[candidate_key_columns[name]])
                else:
                    #print('joining here')
                    dataset.set_index(base_key, inplace=True)
                    #augmented_dataset = augmented_dataset.join(dataset, how='left', lsuffix='_x', rsuffix='_y')  
                    augmented_dataset = pd.merge(augmented_dataset,#.set_index(base_key), 
                                                 dataset, #x.set_index(base_key),
                                                 how='left',
                                                 on=base_key,
                                                 #lsuffix='_x',
                                                 #rsuffix='_y',
                                                 validate='m:1')
                    #print('done joining dataset', name)
        except (pd.errors.EmptyDataError, KeyError, ValueError) as e:
            print('there was an error for dataset', name, e)
            continue

    if prepruning == 'containment':
        #print('all containments')
        #print([elem for elem in sorted(containments.items(), key= lambda x: x[1], reverse=True)])
        if topN:
            chosen_candidates = [elem[0] for elem in sorted(containments.items(), key= lambda x: x[1], reverse=True)[:topN]]
        else:
            if percentage < 1:
                chosen_candidates = [elem[0] for elem in sorted(containments.items(), key= lambda x: x[1], reverse=True)[:int((1.0 - percentage)*len(containments.items()))]]
            else:
                chosen_candidates = [elem[0] for elem in sorted(containments.items(), key= lambda x: x[1], reverse=True)[:percentage]]
        #print('&&&& initial', len(containments.items()), 'final', len(chosen_candidates))
        for name in chosen_candidates:
            #print('here is candidate', name)
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
            
                ### Step 3: augment the table
                if candidate_key_columns:
                    augmented_dataset = pd.merge(augmented_dataset,
                                                 dataset,
                                                 how='left',
                                                 left_on=[base_key],
                                                 right_on=[candidate_key_columns[name]])
                else:
                    dataset.set_index(base_key, inplace=True)
                    augmented_dataset = pd.merge(augmented_dataset,#.set_index(base_key), 
                                                 dataset, #x.set_index(base_key),
                                                 how='left',
                                                 on=base_key,
                                                 validate='m:1')
                
            except (pd.errors.EmptyDataError, KeyError, ValueError):
                print('there was an error for dataset', name, e)
                continue
    
    #augmented_dataset = augmented_dataset.set_index(base_key)
    augmented_dataset = augmented_dataset.select_dtypes(include=['int64', 'float64'])
    augmented_dataset = augmented_dataset.replace([np.inf, -np.inf], np.nan)
    #augmented_dataset.columns = augmented_dataset.columns.str.rstrip('_x')
    print('augmented data shape', augmented_dataset.shape)
    if mean_data_imputation:
        mean = augmented_dataset.mean().replace(np.nan, 0.0)
        #print(mean)
        new_data = augmented_dataset.fillna(mean)
        #fill_NaN = SimpleImputer(missing_values=np.nan, strategy='mean')
        #new_data = pd.DataFrame(fill_NaN.fit_transform(augmented_dataset))
        #print('new data shape', new_data.shape)
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

from sklearn.svm import SVC
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler

FEATURES = ['query_num_of_columns', 'query_num_of_rows', 'query_row_column_ratio', 'query_max_mean', 'query_max_outlier_percentage', 
            'query_max_skewness', 'query_max_kurtosis', 'query_max_unique', 'candidate_num_of_columns', 'candidate_num_rows', 
            'candidate_row_column_ratio', 'candidate_max_mean', 'candidate_max_outlier_percentage', 'candidate_max_skewness', 
            'candidate_max_kurtosis', 'candidate_max_unique', 'query_target_max_pearson', 'query_target_max_spearman', 
            'query_target_max_covariance', 'query_target_max_mutual_info', 'candidate_target_max_pearson', 'candidate_target_max_spearman',
            'candidate_target_max_covariance', 'candidate_target_max_mutual_info', 'containment_fraction']
THETA = 0.7

def train_rbf_svm(features, classes):
    '''
    Builds a model using features to predict associated classes,
    '''
    #print('using rbf')
    feature_scaler = MinMaxScaler().fit(features)
    features_train = feature_scaler.transform(features)
    clf = SVC(max_iter=1000, gamma='auto', probability=True)
    clf.fit(features_train, classes)

    return feature_scaler, clf

from sklearn.ensemble import RandomForestClassifier
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
    

# The next two lines are important for importing files that are in the parent directory, 
# necessary to generate the features
import sys
sys.path.append('../')
from feature_factory import *


def compute_features(query_key_values,
                     candidate_name, 
                     key, 
                     target_name, 
                     augmented_dataset):
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

    candidate_dataset = augmented_dataset.reset_index()[[key, candidate_name]]
    candidate_dataset = candidate_dataset.set_index(key)
    candidate_dataset = candidate_dataset.fillna(candidate_dataset.mean())
    # Step 1: individual candidate features
    #print(candidate_dataset.head())
    feature_factory_candidate = FeatureFactory(candidate_dataset)
    candidate_dataset_individual_features = feature_factory_candidate.get_individual_features(func=max_in_modulus)
    ## For now, we're only using number_of_rows, max_skewness, max_kurtosis, max_number_of_unique_values, 
    ## so we remove the unnecessary elements in the lines below 
    ## candidate_dataset_individual_features = [candidate_dataset_individual_features[index] for index in [1, 5, 6, 7]]

    # Step 2: join the datasets and compute pairwise features IF A NICE, CLEAN AUGMENTED_DATASET WERE NOT PASSED
    # if augmented_dataset.empty:
    #     augmented_dataset = pd.merge(query_dataset, 
    #                                  candidate_dataset,
    #                                  how='left',
    #                                  on=key,
    #                                  validate='m:1')
    #     #augmented_dataset = augmented_dataset.set_index(key)
    #     augmented_dataset = augmented_dataset.replace([np.inf, -np.inf], np.nan).dropna(how="all")
    #     if mean_data_imputation:
    #         #print(augmented_dataset.mean())
    #         #fill_NaN = SimpleImputer(missing_values=np.nan, strategy='mean')
    #         #pd.DataFrame(fill_NaN.fit_transform(augmented_dataset))
    #         new_dataset = augmented_dataset.fillna(augmented_dataset.mean())
    #         new_dataset.columns = augmented_dataset.columns
    #         new_dataset.index = augmented_dataset.index
    #         augmented_dataset = new_dataset
    
    # Step 3: get candidate-target features
    ## The features are, in order: max_query_candidate_pearson, max_query_candidate_spearman, 
    ## max_query_candidate_covariance, max_query_candidate_mutual_info
    column_names = candidate_dataset.columns.tolist() + [target_name]
    feature_factory_candidate_target = FeatureFactory(augmented_dataset[column_names].fillna(augmented_dataset[column_names].mean()))
    candidate_features_target = feature_factory_candidate_target.get_pairwise_features_with_target(target_name, func=max_in_modulus)
     # Step 4: get query-candidate feature "containment ratio". We may not use it in models, but it's 
    ## important to have this value in order to filter candidates in baselines, for example.
    candidate_key_values = candidate_dataset.index.values
    intersection_size = len(set(query_key_values) & set(candidate_key_values))
    containment_ratio = [intersection_size/len(query_key_values)]

    return candidate_dataset_individual_features, candidate_features_target, containment_ratio

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

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


from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split

RANDOM_PREFIX = 'random_feature_'
MAX_ROWS_FOR_ESTIMATION = 5000

def augment_with_random_features(dataset, target_name, number_of_random_features):
    '''
    Given a dataset, the name of the target, and a number of random features, this function derives
    random features that are based on the original ones in the dataset
    '''
    dataset.dropna(inplace=True)
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
    dataset = dataset.replace([np.inf, -np.inf], np.nan)
    dataset = dataset.dropna()
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
    
    augmented_dataset.dropna(inplace=True)
    indices_to_keep = ~augmented_dataset.isin([np.nan, np.inf, -np.inf]).any(1)
    augmented_dataset = augmented_dataset[indices_to_keep].astype(np.float64)
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
        if not selected:
            break
        linreg.fit(X_train[selected], y_train)
        y_pred = linreg.predict(X_test[selected])
        new_r2_score = r2_score(y_test, y_pred)
        if  new_r2_score > current_r2_score:
            current_r2_score = new_r2_score
        else:
            break
    return selected


def compute_model_performance(dataset, 
                              target_name, 
                              features):
    '''
    This function computes the performance of the original regression model the user 
    is interested in for a specific set of features. 
    '''
    
    # Now let's split the data
    X_train, X_test, y_train, y_test = train_test_split(dataset.drop([target_name], axis=1),
                                                        dataset[target_name],
                                                        test_size=0.33,
                                                        random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train[features], y_train.ravel())
    y_pred = model.predict(X_test[features])
    print('R2-score', r2_score(y_test, y_pred))
    
def prune_candidates_with_ida(training_data,
                              augmented_dataset,
                              base_dataset,
                              target_name,
                              key,
                              percentage=0.5):
    '''
    This function trains and uses IDA as a pruner of candidates for augmentation.
    It keeps the top percentage (indicated by parameter percentage) of candidates.
    '''
    
    #Let's train our IDA model over the training dataset
    time1 = time.time()
    feature_scaler, model = train_random_forest(training_data[FEATURES], training_data['class_pos_neg']) 
    #feature_scaler, model = train_rbf_svm(training_data[FEATURES], training_data['class_pos_neg'])
    time2 = time.time()
    print('time to train our model', (time2-time1)*1000.0, 'ms')
    
    #Generate a label for every feature in the augmented dataset
    time1 = time.time()
    candidate_names = set(augmented_dataset.columns) - set(base_dataset.columns)
    ## We just need to compute query features once!
    ## In order, the returned features are number_of_columns, number_of_rows, row_to_column_ratio,
    ## max_mean, max_outlier_percentage, max_skewness, max_kurtosis, max_number_of_unique_values.
    ## For now, we're only using number_of_columns, number_of_rows, row_to_column_ratio, 
    ## max_skewness, max_kurtosis, max_number_of_unique_values, so we remove the unnecessary elements 
    ## in the lines below

    feature_factory_query = FeatureFactory(base_dataset.drop([target_name], axis=1))
    query_features = feature_factory_query.get_individual_features(func=max_in_modulus)
    #query_features = [query_features[index] for index in [0, 1, 2, 5, 6, 7]]
        
    ## get query-target features 
    ## The features are, in order: max_query_target_pearson, max_query_target_spearman, 
    ## max_query_target_covariance, max_query_target_mutual_info
    feature_factory_full_query = FeatureFactory(base_dataset)
    query_features_target = feature_factory_full_query.get_pairwise_features_with_target(target_name, 
                                                                                         func=max_in_modulus)
    query_key_values = base_dataset.index.values
    feature_vectors = []
    #candidate_columns = list(set(augmented_dataset.columns.tolist()) - set(base_dataset.columns.tolist()))
    for name in candidate_names:
        #candidate_dataset = augmented_dataset.reset_index()[[key, name]]
        #print(name, candidate_columns[index])
        candidate_features, candidate_features_target, containment_ratio = compute_features(query_key_values,
                                                                                            name, 
                                                                                            key, 
                                                                                            target_name, 
                                                                                            augmented_dataset)
        feature_vectors.append(query_features + candidate_features + query_features_target + candidate_features_target + containment_ratio)
    predictions = model.predict(normalize_features(np.array(feature_vectors))) 
    gain_pred_probas = [elem[0] for elem in model.predict_proba(normalize_features(np.array(feature_vectors)))]
    probs_dictionary = {name: prob for name, prob in zip(candidate_names, list(gain_pred_probas))}
    pruned = sorted(probs_dictionary.items(), key = lambda x:x[1], reverse=True)[:int((1.0 - percentage)*len(probs_dictionary.items()))]
    candidates_to_keep = [elem[0] for elem in pruned if elem[1] > 0.5] # if elem[1] > 0.5, it was classified as 'keepable'
    time2 = time.time()
    print('time to predict what candidates to keep', (time2-time1)*1000.0, 'ms')
    print('initial number of candidates', len(candidate_names), 'final number of candidates', len(candidates_to_keep))
    return candidates_to_keep    


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

def compute_user_model_performance(dataset, target_name, features, model_type='random_forest'):
    '''
    This function checks how well a random forest (assumed to be the user's model), 
    trained on a given set of features, performs in the prediction of a target
    '''

    time1 = time.time()
    # Now let's split the data
    X_train, X_test, y_train, y_test = train_test_split(dataset.drop([target_name], axis=1),
                                                        dataset[target_name],
                                                        test_size=0.33,
                                                        random_state=42)
    if model_type == 'random_forest':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train[features], y_train.ravel())
        y_pred = model.predict(X_test[features])
    elif model_type == 'linear_regression':
        model = LinearRegression()
        model.fit(X_train[features], y_train.ravel())
        y_pred = model.predict(X_test[features])
    else:
        print('Specified user model is not implemented')
        exit()
    time2 = time.time()
    print('time to create user\'s model with chosen candidates', (time2-time1)*1000.0, 'ms')
    print('R2-score of user model', r2_score(y_test, y_pred))
    #print('MAE of user model', mean_absolute_error(y_test, y_pred))
    #print('MSE of user model', mean_squared_error(y_test, y_pred))
    
from sklearn.feature_selection import RFE
def stepwise_selection(data, target):
    print('USING STEPWISE_SELECTION')
    estimator = LinearRegression()
    selector = RFE(estimator)
    print('before selector')
    reduced = selector.fit(data, target)
    print('after selector')
    return [elem for elem, label in zip(list(data.columns), list(reduced.support_)) if label]


import random
def check_efficiency_with_ida(base_dataset, 
                              key, 
                              target_name, 
                              training_data, 
                              thresholds_tau=[0.1, 0.2, 0.4, 0.6, 0.8], #REFACTOR: this parameter is only used by RIFS
                              eta=0.2, #REFACTOR: this parameter is only used by RIFS
                              k_random_seeds=[42, 17, 23, 2, 5, 19, 37, 41, 13, 33], #REFACTOR: this parameter is only used by RIFS
                              mean_data_imputation=True, 
                              rename_numerical=True, 
                              separator=',',
                              augmented_dataset=None, 
                              feature_selector=wrapper_algorithm,
                              gain_prob_threshold=0.5,
                              path_to_candidates=None,
                              prepruning='containment', 
                              percentage=0.5, 
                              candidate_key_columns=None,
                              topN=None):
    '''
    This function gets the time to run a feature selector without pre-pruning 
    or with pre-pruning using either IDA or a pruning baseline
    '''
    print('Initial performance')
    #print('***', base_dataset.columns)
    compute_user_model_performance(base_dataset, target_name, base_dataset.drop([target_name], axis=1).columns)
    print('******* PREPRUNING STRATEGY ********', prepruning)
    
    #Step 2: let's see how much time it takes to run chosen pre-pruner
    if prepruning == 'ida':
        candidates_to_keep = prune_candidates_with_ida(training_data, 
                                                       augmented_dataset, 
                                                       base_dataset, 
                                                       target_name, 
                                                       key, 
                                                       percentage=percentage)
        
        pruned_dataset = augmented_dataset[base_dataset.columns.to_list() + candidates_to_keep]
        #print('candidates kept by ida', base_dataset.drop([key], axis=1).columns.to_list() + candidates_to_keep)
    elif prepruning == 'none':
        pruned_dataset = augmented_dataset
    elif prepruning == 'containment':
        # if the prepruning is 'containment', the pruning has to be done in the augmentation itself
        pruned_dataset = join_datasets(base_dataset,
                                       path_to_candidates,
                                       key,
                                       separator=separator,
                                       percentage=percentage,
                                       prepruning=prepruning,
                                       topN=topN)
        augmented_dataset = pruned_dataset
        #print('Done creating the pruned dataset', pruned_dataset.shape, pruned_dataset.index)
    elif prepruning == 'random':
        # if the prepruning is random, it will select sqrt(len(candidate_features)) features at random
        candidate_features = set(augmented_dataset.columns.to_list()) - set(base_dataset.columns.to_list()) 
        candidates_to_keep = random.sample(candidate_features, int((1.0 - percentage)*len(candidate_features)))
        pruned_dataset = augmented_dataset[base_dataset.columns.to_list() + candidates_to_keep]
        
    #Step 3: select features with selector over pruned dataset (if RIFS, we inject 20% of random features)
    time1 = time.time()
    if feature_selector == wrapper_algorithm:
        selected_pruned = wrapper_algorithm(pruned_dataset,  
                                            target_name, 
                                            key, 
                                            thresholds_tau, 
                                            eta, 
                                            k_random_seeds)
        #print('selected by rifs', selected_pruned)
    elif feature_selector == boruta_algorithm:
        selected_pruned = boruta_algorithm(pruned_dataset, target_name)
    elif feature_selector == stepwise_selection:
        selected_pruned = stepwise_selection(pruned_dataset.drop([target_name], axis=1), pruned_dataset[target_name])
    elif feature_selector == recursive_feature_elimination:
        selected_pruned = recursive_feature_elimination(pruned_dataset.drop([target_name], axis=1), pruned_dataset[target_name])
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
                                       target_name, 
                                       selected_pruned)
        time2 = time.time()
        print('time to create and assess user\'s model with pruner', prepruning, (time2-time1)*1000.0, 'ms')
    
        
    #print('size of entire dataset', augmented_dataset.shape[1], 'size of pruned', pruned.shape[1])
    #print('size of selected features when you use prepruner', prepruning, len(selected_pruned))
    #return selected_all, candidates_to_keep, selected_pruned, model, probs_dictionary

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

def compute_user_model_performance(dataset, target_name, features, model_type='random_forest'):
    '''
    This function checks how well a random forest (assumed to be the user's model), 
    trained on a given set of features, performs in the prediction of a target
    '''

    time1 = time.time()
    # Now let's split the data
    dataset.dropna(inplace=True)
    indices_to_keep = ~dataset.isin([np.nan, np.inf, -np.inf]).any(1)
    dataset = dataset[indices_to_keep]#.astype(np.float64)
    X_train, X_test, y_train, y_test = train_test_split(dataset.drop([target_name], axis=1),
                                                        dataset[target_name],
                                                        test_size=0.33,
                                                        random_state=42)
    if model_type == 'random_forest':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train[features], y_train.ravel())
        y_pred = model.predict(X_test[features])
    elif model_type == 'linear_regression':
        model = LinearRegression()
        model.fit(X_train[features], y_train.ravel())
        y_pred = model.predict(X_test[features])
    else:
        print('Specified user model is not implemented')
        exit()
    time2 = time.time()
    print('time to create user\'s model with chosen candidates', (time2-time1)*1000.0, 'ms')
    print('R2-score of user model', r2_score(y_test, y_pred))
    print('MAE of user model', mean_absolute_error(y_test, y_pred))
    print('MSE of user model', mean_squared_error(y_test, y_pred))
    
        
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
    #augmented_dataset = augmented_dataset.loc[:,~augmented_dataset.columns.duplicated()]
    candidate_names = set(augmented_dataset.columns) - set(base_dataset.columns)
    feature_vectors = []
    labels = []
    improvements = {}
    feature_factory_query = FeatureFactory(base_dataset.drop([target_name], axis=1))
    query_features = feature_factory_query.get_individual_features(func=max_in_modulus)
    query_features = [query_features[index] for index in [0, 1, 2, 5, 6, 7]]
    feature_factory_full_query = FeatureFactory(base_dataset)
    query_features_target = feature_factory_full_query.get_pairwise_features_with_target(target_name, 
                                                                                         func=max_in_modulus)
    query_key_values = base_dataset.index.values
    for name in candidate_names:
        candidate_dataset = augmented_dataset.reset_index()[[key, name]]
        candidate_features, candidate_features_target, containment_ratio = compute_features(query_key_values,
                                                                                            candidate_dataset.set_index(key), 
                                                                                            key, 
                                                                                            target_name, 
                                                                                            augmented_dataset=augmented_dataset)
        feature_vectors.append(query_features + candidate_features + query_features_target + candidate_features_target)
        
        initial, final, improvement = compute_model_performance_improvement(base_dataset, 
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

PRUNERS = ['ida', 'containment', 'random', 'none']

if __name__ == '__main__':
    '''
    Given a base table, a path to candidates, the name of the key and of the 
    target, this script shows efficiency and effectiveness values for different 
    percentages of pruning.
    '''
    
    path_to_base_table = sys.argv[1]
    path_to_candidates = sys.argv[2]
    key = sys.argv[3]
    target = sys.argv[4]

    openml_training = pd.read_csv('../classification/training-simplified-data-generation.csv')
    openml_training['class_pos_neg'] = ['gain' if row['gain_in_r2_score'] > 0 else 'loss'
                                        for index, row in openml_training.iterrows()]
    openml_training_high_containment = openml_training.loc[openml_training['containment_fraction'] >= THETA]


    base_table = pd.read_csv(path_to_base_table).set_index(key)
    #Step 1: do the join with every candidate dataset in dataset_directory. 
    ## This has to be done both with and without prepruners.
    ## If the prepruning is containment, we already "prune" the augmented dataset while creating it

    
    print('query', path_to_base_table)

    print('top-100')
    check_efficiency_with_ida(base_table,
                              key, 
                              target,
                              openml_training_high_containment,
                              rename_numerical=True,
                              separator=',',
                              path_to_candidates=path_to_candidates,
                              prepruning='containment',
                              #feature_selector=stepwise_selection,
                              topN=38)

    print('top-50')
    check_efficiency_with_ida(base_table,
                              key, 
                              target,
                              openml_training_high_containment,
                              rename_numerical=True,
                              path_to_candidates=path_to_candidates,
                              separator=',',
                              prepruning='containment',
                              #feature_selector=stepwise_selection,
                              topN=38)
    print('top-20')
    check_efficiency_with_ida(base_table,
                              key, 
                              target,
                              openml_training_high_containment,
                              rename_numerical=True,
                              path_to_candidates=path_to_candidates,
                              separator=',',
                              #feature_selector=stepwise_selection,
                              prepruning='containment',
                              topN=18)

    print('top-10')
    check_efficiency_with_ida(base_table,
                              key, 
                              target,
                              openml_training_high_containment,
                              rename_numerical=True,
                              path_to_candidates=path_to_candidates,
                              separator=',',
                              #feature_selector=stepwise_selection,
                              prepruning='containment',
                              topN=8)

    print('top-5')
    check_efficiency_with_ida(base_table,
                              key, 
                              target,
                              openml_training_high_containment,
                              rename_numerical=True,
                              path_to_candidates=path_to_candidates,
                              separator=',',
                              prepruning='containment',
                              #feature_selector=stepwise_selection,
                              topN=3)

    print('top-3')
    check_efficiency_with_ida(base_table,
                              key, 
                              target,
                              openml_training_high_containment,
                              rename_numerical=True,
                              path_to_candidates=path_to_candidates,
                              separator=',',
                              #feature_selector=stepwise_selection,
                              prepruning='containment',
                              topN=1)

    print('top-1')
    check_efficiency_with_ida(base_table,
                              key, 
                              target,
                              openml_training_high_containment,
                              rename_numerical=True,
                              path_to_candidates=path_to_candidates,
                              separator=',',
                              #feature_selector=stepwise_selection,
                              prepruning='containment',
                              topN=1)
