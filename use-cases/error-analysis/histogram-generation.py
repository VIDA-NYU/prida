""" Given (1) a use case dataset with features FEATURES, target TARGET, and column eval indicating whether each instance/row is  a 
          false positive (FP), true positive (TP), false negative (FN), or true negative (TN) according to a certain classifier

          this script generates histograms for every feature and target separated by (a) FP, TP, FN, and TN, or (b) "gain" and "loss".
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import median_absolute_deviation

OUTLIER_THRESHOLD_ZSCORES = 3
OUTLIER_THRESHOLD_MAD = 2
TARGET = 'gain_in_r2_score'
ALPHA = 0
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

def split_augmentations_into_gain_and_loss(data):
  """This function splits instances into those that correspond
  to positive augmentations (gain) and negative augmentations (loss).

  The criterion is a simple, global one: if the value for column TARGET
  is above a certain threshold ALPHA, the augmentation is positive; otherwise, it 
  is negative
  """
  positive = data.loc[data[TARGET] > ALPHA]
  negative = data.loc[data[TARGET] <= ALPHA]
  return positive, negative
  
if __name__=='__main__':
  use_case_dataset = pd.read_csv(sys.argv[1])
  
  # fp = use_case_dataset.loc[use_case_dataset['eval'] == 'fp']
  # plot_features_and_target_histograms(fp, 'fp')
  # tp = use_case_dataset.loc[use_case_dataset['eval'] == 'tp']
  # plot_features_and_target_histograms(tp, 'tp')
  # fn = use_case_dataset.loc[use_case_dataset['eval'] == 'fn']
  # plot_features_and_target_histograms(fn, 'fn')
  # tn = use_case_dataset.loc[use_case_dataset['eval'] == 'tn']
  # plot_features_and_target_histograms(tn, 'tn')
    
  gain, loss = split_augmentations_into_gain_and_loss(use_case_dataset)
  plot_features_and_target_histograms(gain, 'gain')
  plot_features_and_target_histograms(loss, 'loss')
