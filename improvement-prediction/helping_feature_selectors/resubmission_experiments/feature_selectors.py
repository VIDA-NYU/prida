'''
Here, we implement different feature selectors
'''

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.feature_selection import RFE, SelectKBest, f_regression, mutual_info_regression
from scipy.stats import pearsonr

# =============== RIFS ================= #

RANDOM_PREFIX = 'random_feature_'
MAX_ROWS_FOR_ESTIMATION = 5000
THRESHOLDS_TAU = [0.1, 0.2, 0.4, 0.6, 0.8]
ETA = 0.2
K_RANDOM_SEEDS = [42, 17, 23, 2, 5, 19, 37, 41, 13, 33]

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
    #print(feats_quality)
    sorted_feats =  sorted(feats_quality.items(), 
                           key=lambda x: x[1], 
                           reverse=True)
    return [(elem[0], elem[1]/len(rankings)) for elem in sorted_feats]

def random_injection_feature_selection(augmented_dataset,
                                       target_name,
                                       tau):
    '''
    This is the core of ARDA's feature selection algorithm RIFS. Given an augmented dataset, ideally 
    created through joins over sketches over the original datasets, a threshold tau for 
    quality of features, a fraction eta of random features to inject, and k random seeds to perform 
    k experiments in a reproducible way, it selects the features that should be used in the augmented 
    dataset
    '''
    number_of_random_features = int(np.ceil(ETA*augmented_dataset.drop([target_name], axis=1).shape[1]))
    augmented_dataset_with_random = augment_with_random_features(augmented_dataset,
                                                                 target_name, 
                                                                 number_of_random_features)
    
    # Now we obtain rankings using random forests and sparse regression models
    ## the paper does not say what hyperparameters were used in the experiment
    rankings = []
    for seed in K_RANDOM_SEEDS:
        #print('seed', seed)
        rf = RandomForestRegressor(n_estimators=100, random_state=seed)
        rf.fit(augmented_dataset_with_random.drop([target_name], axis=1), augmented_dataset_with_random[target_name])
        rf_coefs = rf.feature_importances_        
        ## This version of lasso is giving lower weights to random features, which is good
        lasso = Lasso(random_state=seed)
        lasso.fit(augmented_dataset_with_random.drop([target_name], axis=1), augmented_dataset_with_random[target_name])
        lasso_coefs = lasso.coef_
        rank = combine_rankings(rf_coefs, lasso_coefs, augmented_dataset_with_random.drop([target_name], axis=1).columns)
        rankings.append(rank)
    
    # Now, for each non-random feature, we get the number of times it appeared in front of 
    ## all random features
    sorted_features = aggregate_features_by_quality(rankings)
    #print('sorted by quality', sorted_features)
    return [elem[0] for elem in sorted_features if elem[1] >= tau]

def rifs(augmented_dataset, target_name, key):
    '''
    This function implements ARDA's feature selection algorithm (RIFS), which searches 
    for the best subset of features by doing an exponential search
    '''

    print('USING RIFS')
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
    for tau in THRESHOLDS_TAU:
        #print('tau', tau)
        selected = random_injection_feature_selection(augmented_dataset,
                                                      target_name, 
                                                      tau)
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

# ====================== OTHER SELECTORS ===================== #

def recursive_feature_elimination(data, target):
    print('USING RECURSIVE FEATURE ELIMINATION')
    estimator = LinearRegression()
    selector = RFE(estimator)
    #data.dropna(inplace=True)
    mean = data.mean().replace(np.nan, 0.0)
    data = data.fillna(mean)
    #indices_to_keep = ~data.isin([np.nan, np.inf, -np.inf]).any(1)
    #data = data[indices_to_keep].astype(np.float64)
    reduced = selector.fit(data.replace(np.nan, 0.0), target)
    return [elem for elem, label in zip(list(data.columns), list(reduced.support_)) if label]

def select_f_regression(data, target):
    print('USING F_REGRESSION')
    selector = SelectKBest(score_func=f_regression, k=int(np.sqrt(len(list(data.columns)))))
    selector.fit(data.replace(np.nan, 0.0), target)
    cols = selector.get_support(indices=True)
    new_data = data.iloc[:,cols]
    return new_data.columns.tolist()

def select_based_on_correlation(data, target):
    print('USING CORRELATION')
    correlations = {}
    for attribute in data.columns:
        try:
            corr, pvalue = pearsonr(np.array(data[attribute].replace(np.nan, 0.0)), np.array(target))
            
            if not np.isnan(corr):# != float('nan'):
                correlations[attribute] = corr
        except:
            continue
        
    columns = [elem[0] for elem in sorted(correlations.items(), key =lambda x: np.fabs(x[1]), reverse=True)][:int(np.sqrt(len(correlations)))]
    return columns

def select_mutual_info_regression(data, target):
    print('USING MUTUAL INFO REGRESSION')
    selector = SelectKBest(score_func=mutual_info_regression, k=int(np.sqrt(len(list(data.columns)))))
    selector.fit(data.replace(np.nan, 0.0), target)
    cols = selector.get_support(indices=True)
    new_data = data.iloc[:,cols]
    return new_data.columns.tolist()
