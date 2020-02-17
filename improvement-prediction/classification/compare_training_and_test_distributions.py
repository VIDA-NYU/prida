'''
training and test files have metadata, learning features, and target 
values for each line
'''


FIRST_FEATURE_ID = 3
FIRST_TARGET_ID = -4
FEATURE_NAMES = ['query_num_of_columns','query_num_of_rows','query_row_column_ratio','query_max_mean','query_max_outlier_percentage','query_max_skewness','query_max_kurtosis','query_max_unique','candidate_num_of_columns','candidate_num_rows','candidate_row_column_ratio','candidate_max_mean','candidate_max_outlier_percentage','candidate_max_skewness','candidate_max_kurtosis','candidate_max_unique','query_target_max_pearson','query_target_max_spearman','query_target_max_covariance','query_target_max_mutual_info','candidate_target_max_pearson','candidate_target_max_spearman','candidate_target_max_covariance','candidate_target_max_mutual_info','max_pearson_difference', 'containment_fraction']
OUTLIER_THRESHOLD_ZSCORES = 3
OUTLIER_THRESHOLD_MAD = 2
GAIN_THRESHOLD = 0.05

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

def plot_histogram(hist_filename, feature_name, feature_training, feature_test, remove_outliers_zscores=False, remove_outliers_mad=False):
    if remove_outliers_zscores:
        feature_training = remove_outliers_based_on_zscores(feature_training)
        feature_test = remove_outliers_based_on_zscores(feature_test)
    elif remove_outliers_mad:
        feature_training = remove_outliers_based_on_mad(feature_training)
        feature_test = remove_outliers_based_on_mad(feature_test)

    weights = np.ones_like(feature_training)/float(len(feature_training))
    plt.hist(feature_training, bins=50, alpha=0.5, weights=weights, label='positive-gain')#, normed=True) #density=True, stacked=True)

    weights = np.ones_like(feature_test)/float(len(feature_test))
    plt.hist(feature_test, bins=50, alpha=0.5, weights=weights, label='negative-gain')#, normed=True) #density=True, stacked=True) #, density=True
         
    plt.legend(loc='upper right')
    plt.xlabel('Value Ranges')
    plt.ylabel('Percentages')
    plt.title(feature_name)
    plt.savefig(hist_filename,  dpi=600)
    plt.close()

def separate_feature_based_on_gain_range(feature, gains):
    '''
    this function splits one feature into two arrays: one where the corresponding gains are above 
    GAIN_THRESHOLD and another where they are below -GAIN_THRESHOLD
    '''

    positive = []
    negative = []
    for value, gain in zip(feature, gains):
        if gain > GAIN_THRESHOLD:
            positive.append(value)
        elif gain < -GAIN_THRESHOLD:
            negative.append(value)
    return positive, negative

if __name__ == '__main__':
    training_file = sys.argv[1]
    test_file = sys.argv[2]

    training_lines = open(sys.argv[1]).readlines()[1:]
    test_lines = open(sys.argv[2]).readlines()[1:]
    
    training = []
    training_gains = []
    for line in training_lines:
        fields = line.strip().split(',')
        features = [float(i) for i in fields[FIRST_FEATURE_ID:FIRST_TARGET_ID]]
        training.append(features)
        training_gains.append(float(fields[-1]))
    training = np.array(training)
    
    test = []
    test_gains = []
    for line in test_lines:
        fields = line.strip().split(',')
        features = [float(i) for i in fields[FIRST_FEATURE_ID:FIRST_TARGET_ID]]
        test.append(features)
        test_gains.append(float(fields[-1]))
    test = np.array(test)

    if training.shape[1] != test.shape[1]:
        print('Training and test data have different numbers of features. Please correct this and run this script again')
        exit()

#    num_features = training.shape[1]   
#    for i in range(num_features):
#        plot_histogram(FEATURE_NAMES[i] + '_histograms_training_test.png', FEATURE_NAMES[i], training[:,i], test[:,i], remove_outliers_mad=True) 
#    plot_histogram(FEATURE_NAMES[8] + '_histograms_training_test.png', FEATURE_NAMES[8], training[:,8], test[:,8])
#    plot_histogram(FEATURE_NAMES[12] + '_histograms_training_test.png', FEATURE_NAMES[12], training[:,12], test[:,12])

    plot_histogram('gains_in_r2_score_histograms_training_test.png', 'gains_in_r2_score', training_gains, test_gains, remove_outliers_mad=True)

    num_features = training.shape[1]   
    for i in range(num_features):
        feature_for_positive_gain, feature_for_negative_gain = separate_feature_based_on_gain_range(training[:,i], training_gains) 
        plot_histogram(FEATURE_NAMES[i] + '_training_corresponding_positive_and_negative_gains.png', FEATURE_NAMES[i], feature_for_positive_gain, feature_for_negative_gain, remove_outliers_mad=True)
        feature_for_positive_gain, feature_for_negative_gain = separate_feature_based_on_gain_range(test[:,i], test_gains) 
        plot_histogram(FEATURE_NAMES[i] + '_test_corresponding_positive_and_negative_gains.png', FEATURE_NAMES[i], feature_for_positive_gain, feature_for_negative_gain, remove_outliers_mad=True) 
