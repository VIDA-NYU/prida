'''
training and test files have metadata, learning features, and target 
values for each line
'''


FIRST_FEATURE_ID = 3
FIRST_TARGET_ID = -4
FEATURE_NAMES = ['query_num_of_columns','query_num_of_rows','query_row_column_ratio','query_max_mean','query_max_outlier_percentage','query_max_skewness','query_max_kurtosis','query_max_unique','candidate_num_of_columns','candidate_num_rows','candidate_row_column_ratio','candidate_max_mean','candidate_max_outlier_percentage','candidate_max_skewness','candidate_max_kurtosis','candidate_max_unique','query_target_max_pearson','query_target_max_spearman','query_target_max_covariance','query_target_max_mutual_info','candidate_target_max_pearson','candidate_target_max_spearman','candidate_target_max_covariance','candidate_target_max_mutual_info','max_pearson_difference', 'containment_fraction']
OUTLIER_THRESHOLD_ZSCORES = 3
OUTLIER_THRESHOLD_MAD = 2

import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy.stats import median_absolute_deviation

def remove_outliers_based_on_zscores(data):
    mean_ = np.mean(data)
    std_ = np.std(data)
    return [i for i in data if np.fabs((i - mean_)/std_) < OUTLIER_THRESHOLD_ZSCORES]

def remove_outliers_based_on_mad(data):
    mad = median_absolute_deviation(data)
    median = np.median(data)
    return [i for i in data if np.fabs((i - median)/mad) < OUTLIER_THRESHOLD_MAD]

def plot_histogram(feature_name, feature_training, feature_test, remove_outliers_zscores=False, remove_outliers_mad=False):
    if remove_outliers_zscores:
        feature_training = remove_outliers_based_on_zscores(feature_training)
        feature_test = remove_outliers_based_on_zscores(feature_test)
    elif remove_outliers_mad:
        feature_training = remove_outliers_based_on_mad(feature_training)
        feature_test = remove_outliers_based_on_mad(feature_test)
    
    x, bins1, p1 = plt.hist(feature_training, bins=50, density=True, alpha=0.5, label='training')

    y, bins2, p2 = plt.hist(feature_test, bins=50, density=True, alpha=0.5, label='test')
         
    plt.legend(loc='upper right')
    plt.title(feature_name)
    plt.savefig(feature_name + '_histograms_training_test.png', dpi=600)
    plt.close()
    
if __name__ == '__main__':
    training_file = sys.argv[1]
    test_file = sys.argv[2]

    training_lines = open(sys.argv[1]).readlines()
    test_lines = open(sys.argv[2]).readlines()
    
    training = []
    for line in training_lines:
        fields = line.strip().split(',')
        features = [float(i) for i in fields[FIRST_FEATURE_ID:FIRST_TARGET_ID]]
        training.append(features)
    training = np.array(training)
    
    test = []
    for line in test_lines:
        fields = line.strip().split(',')
        features = [float(i) for i in fields[FIRST_FEATURE_ID:FIRST_TARGET_ID]]
        test.append(features)
    test = np.array(test)

    if training.shape[1] != test.shape[1]:
        print('Training and test data have different numbers of features. Please correct this and run this script again')
        exit()

    plot_histogram(FEATURE_NAMES[8], training[:,8], test[:,8])#, remove_outliers_zscores=False, remove_outliers_mad=True)
    plot_histogram(FEATURE_NAMES[12], training[:,12], test[:,12])
    #num_features = training.shape[1]   
    #for i in range(num_features):
    #    plot_histogram(FEATURE_NAMES[i], training[:,i], test[:,i], remove_outliers_mad=True) 

