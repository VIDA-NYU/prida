'''
This script gets a file with metadata, learning features, and target 
values for each line, and plots different features against the target.

'''

FEATURE_NAMES = ['query_num_of_columns','query_num_of_rows','query_row_column_ratio','query_max_mean','query_max_outlier_percentage','query_max_skewness','query_max_kurtosis','query_max_unique','candidate_num_of_columns','candidate_num_rows','candidate_row_column_ratio','candidate_max_mean','candidate_max_outlier_percentage','candidate_max_skewness','candidate_max_kurtosis','candidate_max_unique','query_target_max_pearson','query_target_max_spearman','query_target_max_covariance','query_target_max_mutual_info','candidate_target_max_pearson','candidate_target_max_spearman','candidate_target_max_covariance','candidate_target_max_mutual_info','max_pearson_difference', 'containment_fraction']
TARGET_GAIN_IN_R2_SCORE_ID = -1
FIRST_FEATURE_ID = 3
FIRST_TARGET_ID = -4
OUTLIER_THRESHOLD_MAD = 2
OUTLIER_THRESHOLD_ZSCORES = 3

import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy.stats import median_absolute_deviation

def normalize(data):
    max_ = max(data)
    return [i/max_ for i in data]

def remove_outliers_based_on_zscores(x_data, y_data):
    mean_x = np.mean(x_data)
    std_x = np.std(x_data)
    mean_y = np.mean(y_data)
    std_y = np.std(y_data)
    filtered_x = []
    filtered_y = []
    for x, y in zip(x_data, y_data):
        if np.fabs((x - mean_x)/std_x) < OUTLIER_THRESHOLD_ZSCORES and np.fabs((y - mean_y)/std_y) < OUTLIER_THRESHOLD_ZSCORES:
            filtered_x.append(x)
            filtered_y.append(y)
    return filtered_x, filtered_y

def remove_outliers_based_on_mad(x_data, y_data):
    mad_x = median_absolute_deviation(x_data)
    median_x = np.median(x_data)
    mad_y = median_absolute_deviation(y_data)
    median_y = np.median(y_data)
    filtered_x = []
    filtered_y = []
    for x, y in zip(x_data, y_data):
        if np.fabs((x - median_x)/mad_x) < OUTLIER_THRESHOLD_MAD and np.fabs((y - median_y)/mad_y) < OUTLIER_THRESHOLD_MAD:
            filtered_x.append(x)
            filtered_y.append(y)
    return filtered_x, filtered_y

def plot_scatterplot(feature_data, target_data, image_name, xlabel, ylabel):
    feature_data, target_data = remove_outliers_based_on_zscores(feature_data, target_data)
    if not feature_data or not target_data:
        return
    #plt.scatter(normalize(feature_data), normalize(target_data), alpha=0.5)
    plt.scatter(feature_data, target_data, alpha=0.5)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(image_name, dpi=300)
    plt.close()


if __name__ == '__main__':
    filename = sys.argv[1]
    lines = open(sys.argv[1]).readlines()
    
    features = []
    target = []
    for line in lines:
        fields = line.strip().split(',')
        features.append([float(i) for i in fields[FIRST_FEATURE_ID:FIRST_TARGET_ID]])
        target.append(float(fields[TARGET_GAIN_IN_R2_SCORE_ID]))
    features = np.array(features)
    target = np.array(target)
    num_features = features.shape[1]   
    for i in range(num_features):
        plot_scatterplot(features[:,i], target, FEATURE_NAMES[i] + '_vs_gain_in_r2_score.png', FEATURE_NAMES[i], 'gain_in_r2_score')
