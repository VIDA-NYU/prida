""" Given (1) a use case dataset with features FEATURES, target TARGET, and column eval indicating whether each instance/row is  a 
          false positive (FP), true positive (TP), false negative (FN), or true negative (TN) according to a certain classifier

          this script generates histograms for every feature and target separated by FP, TP, FN, and TN.
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import median_absolute_deviation

OUTLIER_THRESHOLD_ZSCORES = 3
OUTLIER_THRESHOLD_MAD = 2
TARGET = 'gain_in_r2_score'
FEATURES = ['query_num_of_columns', 'query_num_of_rows', 'query_row_column_ratio', 'query_max_skewness', 'query_max_kurtosis',
            'query_max_unique', 'candidate_num_of_columns', 'candidate_num_rows', 'candidate_row_column_ratio', 'candidate_max_skewness',
            'candidate_max_kurtosis', 'candidate_max_unique', 'query_target_max_pearson', 'query_target_max_spearman', 'query_target_max_covariance',
            'query_target_max_mutual_info', 'candidate_target_max_pearson', 'candidate_target_max_spearman', 'candidate_target_max_covariance',
            'candidate_target_max_mutual_info', 'max_pearson_difference', 'containment_fraction']

def remove_outliers_based_on_zscores(feature):
  mean_ = np.mean(feature)
  std_ = np.std(feature)
  return [i for i in feature if np.fabs((i - mean_)/std_) < OUTLIER_THRESHOLD_ZSCORES]

def remove_outliers_based_on_mad(feature):
  mad = median_absolute_deviation(feature)
  median = np.median(feature)
  return [i for i in feature if np.fabs((i - median)/mad) < OUTLIER_THRESHOLD_MAD]

def  plot_features_and_target_histograms(data, prefix):
  for feature_name in FEATURES:
    #tmp = remove_outliers_based_on_zscores(data[feature_name])
    #tmp = remove_outliers_based_on_mad(tmp)
    #tmp = remove_outliers_based_on_mad(data[feature_name])
    tmp = data[feature_name]

    weights = np.ones_like(tmp)/float(len(tmp))
    plt.hist(tmp, bins=50, alpha=0.5, weights=weights)
    plt.xlabel('Value Ranges')
    plt.ylabel('Percentages')
    plt.title(feature_name)
    plt.savefig(prefix + '-FEATURE-' + feature_name + '.png',  dpi=600)
    plt.close()

  #tmp = remove_outliers_based_on_zscores(data[TARGET])
  #tmp = remove_outliers_based_on_mad(tmp)
  #tmp = remove_outliers_based_on_mad(data[TARGET])
  tmp = data[feature_name]

  weights = np.ones_like(tmp)/float(len(tmp))
  plt.hist(tmp, bins=50, alpha=0.5, weights=weights)
  plt.xlabel('Value Ranges')
  plt.ylabel('Percentages')
  plt.title(TARGET)
  plt.savefig(prefix + '-TARGET-' + TARGET + '.png',  dpi=600)
  plt.close()


if __name__=='__main__':
  use_case_dataset = pd.read_csv(sys.argv[1])
  
  fp = use_case_dataset.loc[use_case_dataset['eval'] == 'fp']
  plot_features_and_target_histograms(fp, 'fp')
  tp = use_case_dataset.loc[use_case_dataset['eval'] == 'tp']
  plot_features_and_target_histograms(tp, 'tp')
  fn = use_case_dataset.loc[use_case_dataset['eval'] == 'fn']
  plot_features_and_target_histograms(fn, 'fn')
  tn = use_case_dataset.loc[use_case_dataset['eval'] == 'tn']
  plot_features_and_target_histograms(tn, 'tn')
  
    
#    num_features = training.shape[1]   
#    for i in range(num_features):
#        plot_histogram(FEATURE_NAMES[i] + '_histograms_training_test.png', FEATURE_NAMES[i], training[:,i], test[:,i], remove_outliers_mad=True) 
#    plot_histogram(FEATURE_NAMES[8] + '_histograms_training_test.png', FEATURE_NAMES[8], training[:,8], test[:,8])
#    plot_histogram(FEATURE_NAMES[12] + '_histograms_training_test.png', FEATURE_NAMES[12], training[:,12], test[:,12])

    # plot_histogram('gains_in_r2_score_histograms_training_test.png', 'gains_in_r2_score', training_gains, test_gains, remove_outliers_mad=True)

    # num_features = training.shape[1]   
    # for i in range(num_features):
    #     feature_for_positive_gain, feature_for_negative_gain = separate_feature_based_on_gain_range(training[:,i], training_gains) 
    #     plot_histogram(FEATURE_NAMES[i] + '_training_corresponding_positive_and_negative_gains.png', FEATURE_NAMES[i], feature_for_positive_gain, feature_for_negative_gain, remove_outliers_mad=True)
    #     feature_for_positive_gain, feature_for_negative_gain = separate_feature_based_on_gain_range(test[:,i], test_gains) 
    #     plot_histogram(FEATURE_NAMES[i] + '_test_corresponding_positive_and_negative_gains.png', FEATURE_NAMES[i], feature_for_positive_gain, feature_for_negative_gain, remove_outliers_mad=True) 
