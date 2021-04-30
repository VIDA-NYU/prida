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
import warnings; warnings.simplefilter('ignore')

TRAINING_FILENAME = '../classification/training-simplified-data-generation.csv'
THETA = 0.7
SEPARATOR = ','

DATASET_FEATURES = ['query_num_of_columns', 'query_num_of_rows', 'query_row_column_ratio', 'query_max_mean', 'query_max_outlier_percentage', 
                    'query_max_skewness', 'query_max_kurtosis', 'query_max_unique', 'candidate_num_of_columns', 'candidate_num_rows', 
                    'candidate_row_column_ratio', 'candidate_max_mean', 'candidate_max_outlier_percentage']

DATASET_TARGET_FEATURES = ['candidate_max_skewness', 'candidate_max_kurtosis', 'candidate_max_unique', 'query_target_max_pearson', 'query_target_max_spearman', 
                           'query_target_max_covariance', 'query_target_max_mutual_info', 'candidate_target_max_pearson', 'candidate_target_max_spearman',
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

def check_efficiency_and_effectiveness(base_dataset,
                                       path_to_candidates,
                                       key, 
                                       target,
                                       training_data,
                                       rename_numerical=True,
                                       separator=SEPARATOR,
                                       feature_selector=rfe, 
                                       topN=100):
    '''
    This function gets the time to run a feature selector with and without
    pre-pruning using the hierarchical classifier
    '''
    
    print('Initial performance')
    compute_user_model_performance(base_dataset, target_name, base_dataset.drop([target_name], axis=1).columns)
    print('******* PRUNING WITH HIERARCHICAL CLASSIFIER ********')
    #Step 2: let's see how much time it takes to run the classifier-based pruner
    candidates_to_keep = prune_candidates(training_data,  
                                          base_dataset,
                                          path_to_candidates,
                                          key, 
                                          target_name,
                                          topN=topN)
    
    # TODO REFACTOR AND FINISH THIS CODE
    #     pruned_dataset = augmented_dataset[base_dataset.columns.to_list() + candidates_to_keep]
    #     #print('candidates kept by ida', pruned_dataset.columns.to_list())
    # elif prepruning == 'none' or prepruning == 'containment':
    #     # if the prepruning is 'containment', the pruning is already done in the augmentation itself
    #     pruned_dataset = augmented_dataset
    # elif prepruning == 'random':
    #     # if the prepruning is random, it will select sqrt(len(candidate_features)) features at random
    #     candidate_features = set(augmented_dataset.columns.to_list()) - set(base_dataset.columns.to_list()) 
    #     candidates_to_keep = random.sample(candidate_features, int((1.0 - percentage)*len(candidate_features)))
    #     pruned_dataset = augmented_dataset[base_dataset.columns.to_list() + candidates_to_keep]

    # #Step 3: select features with selector over pruned dataset (if RIFS, we inject 20% of random features)
    # time1 = time.time()
    # if feature_selector == wrapper_algorithm:
    #     print('pruned_dataset columns', pruned_dataset.columns.tolist())
    #     selected_pruned = wrapper_algorithm(pruned_dataset,  
    #                                         target_name, 
    #                                         key, 
    #                                         thresholds_tau, 
    #                                         eta, 
    #                                         k_random_seeds)
    #     #print('selected by rifs', selected_pruned)
    # elif feature_selector == boruta_algorithm:
    #     selected_pruned = boruta_algorithm(pruned_dataset, target_name)
    # elif feature_selector == stepwise_selection:
    #     selected_pruned = stepwise_selection(pruned_dataset.drop([target_name], axis=1), pruned_dataset[target_name])
    # elif feature_selector == recursive_feature_elimination:
    #     selected_pruned = recursive_feature_elimination(pruned_dataset.drop([target_name], axis=1), pruned_dataset[target_name])
    # else:
    #     print('feature selector that was passed is not implemented')
    #     exit()
    # time2 = time.time()
    # print('time to run feature selector', (time2-time1)*1000.0, 'ms')

    # #Step 4: compute the user's regression model with features selected_pruned 
    # if len(selected_pruned) == 0:
    #     print('No features were selected. Can\'t run user\'s model.')
    # else:
    #     time1 = time.time()
    #     compute_user_model_performance(augmented_dataset, 
    #                                    target_name, 
    #                                    selected_pruned)
    #     time2 = time.time()
    #     print('time to create and assess user\'s model with pruner', prepruning, (time2-time1)*1000.0, 'ms')

if __name__ == '__main__':    
    path_to_base_table = sys.argv[1]
    path_to_candidates = sys.argv[2]
    key = sys.argv[3]
    target = sys.argv[4]

    openml_training = pd.read_csv(TRAINING_FILENAME)
    openml_training['class_pos_neg'] = ['gain' if row['gain_in_r2_score'] > 0 else 'loss'
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
