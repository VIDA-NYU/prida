'''
Here, we implement a few different feature selection methods Fi, including model-specific approaches such as RIFS and also random forests. 
If we run Fi without preprocessing it with PRIDA, it takes time X and produces Y good attributes (i.e., the performance with the Y attributes 
is better than without them). If we use PRIDA before Fi, we check if the new time X' is significantly lower than X and what percentage of the 
Y attributes is kept. 

argv[1] => base table
argv[2] => candidates' directory
argv[3] => key
argv[4] => target variable
argv[5] => feature selector (e.g., rifs or recursive_feature_elimination)
argv[6] => topN
'''

import sys
sys.path.append('.')
from hierarchical_classifier import *
import functools
   
def perform_feature_selection(base_dataset, 
                              candidate_directory,
                              key,
                              target,
                              training_data, 
                              feature_selector='f_regression', 
                              model_type='random_forest',
                              pruning=False,
                              topN=100):
    '''
    This function augments the base dataset, performs feature selection and 
    assesses the quality of the final model
    '''
                                          
    if not pruning:
        print('########## Performance without PRIDA ###########')
        time1 = time.time()
        candidates = read_candidates(candidate_directory, key)
        augmented_dataset, names_and_columns = join_datasets(base_dataset, candidates, key)
    else:
        print('########## Performance with PRIDA ###########')
        time1 = time.time()
        augmented_dataset = prune_candidates_hierarchical(training_data,  
                                                          base_dataset,
                                                          path_to_candidates,
                                                          key, 
                                                          target,
                                                          topN=topN)
        if sorted(augmented_dataset.columns.tolist()) == sorted(base_dataset.columns.tolist()):
            print('All candidates were pruned and no augmentation was performed')
            exit()
            
    if feature_selector == 'rifs':
        selected_pruned = rifs(augmented_dataset,  
                               target, 
                               key) 
    elif feature_selector == 'recursive_feature_elimination':
        selected_pruned = recursive_feature_elimination(augmented_dataset.drop([target], axis=1), augmented_dataset[target])
    elif feature_selector == 'f_regression':
        selected_pruned = select_f_regression(augmented_dataset.drop([target], axis=1), augmented_dataset[target])
    elif feature_selector == 'correlation':
        selected_pruned = select_based_on_correlation(augmented_dataset.drop([target], axis=1), augmented_dataset[target])
    elif feature_selector == 'mutual_info_regression':
        selected_pruned = select_mutual_info_regression(augmented_dataset.drop([target], axis=1), augmented_dataset[target])
    else:
        print('feature selector that was passed is not implemented')
        exit()

    compute_user_model_performance(augmented_dataset, target, selected_pruned, model_type=model_type)
    time2 = time.time()
    print('time to run feature selector and assess the quality of the new model', (time2-time1)*1000.0, 'ms')

#@functools.lru_cache(maxsize = None)
def check_preprocessing_quality(base_dataset,
                                path_to_candidates,
                                key, 
                                target,
                                training_data,
                                rename_numerical=True,
                                separator=SEPARATOR,
                                topN=100,
                                feature_selector='rifs',
                                user_model_algorithm='random_forest',
                                pruning=False):
    '''
    This function gets the time to run a feature selector with and without
    pre-pruning with prida
    '''
    
    print('########### Initial performance ############')
    compute_user_model_performance(base_dataset, target, base_dataset.drop([key, target], axis=1).columns, model_type=user_model_algorithm)
    perform_feature_selection(base_dataset,
                              path_to_candidates,
                              key,
                              target,
                              training_data, 
                              feature_selector=feature_selector,
                              model_type=user_model_algorithm,
                              pruning=pruning,
                              topN=topN)
                

if __name__ == '__main__':    
    path_to_base_table = sys.argv[1]
    path_to_candidates = sys.argv[2]
    key = sys.argv[3]
    target = sys.argv[4]
    feature_selector = sys.argv[5]
    topN = sys.argv[6]
    
    openml_training = pd.read_csv(TRAINING_FILENAME)
    openml_training[CLASS_ATTRIBUTE_NAME] = ['gain' if row['gain_in_r2_score'] > 0 else 'loss'
                                        for index, row in openml_training.iterrows()]
    openml_training_high_containment = openml_training.loc[openml_training['containment_fraction'] >= THETA]

    base_table = pd.read_csv(path_to_base_table)

    check_preprocessing_quality(base_table,
                                path_to_candidates,
                                key, 
                                target,
                                openml_training_high_containment,
                                rename_numerical=True,
                                separator=SEPARATOR,
                                topN=topN,
                                feature_selector=feature_selector)
    
    #check_preprocessing_quality.cache_clear()

    base_table = pd.read_csv(path_to_base_table)
    check_preprocessing_quality(base_table,
                                path_to_candidates,
                                key, 
                                target,
                                openml_training_high_containment,
                                rename_numerical=True,
                                separator=SEPARATOR,
                                topN=topN,
                                feature_selector=feature_selector,
                                pruning=True)
