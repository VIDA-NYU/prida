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

ABBREVIATIONS = {'query_num_of_columns': 'QC', 'query_num_of_rows': 'QR', 'query_row_column_ratio': 'QRC', 'query_max_skewness': 'QS', 
                 'query_max_kurtosis': 'QK', 'query_max_unique': 'QU', 'candidate_num_of_columns': 'CC', 'candidate_num_rows':'CR', 
                 'candidate_row_column_ratio': 'CRC', 'candidate_max_skewness': 'CS', 'candidate_max_kurtosis': 'CK', 
                 'candidate_max_unique': 'CU', 'query_target_max_pearson': 'QTP', 'query_target_max_spearman': 'QTS', 
                 'query_target_max_covariance': 'QTC', 'query_target_max_mutual_info': 'QTM', 'candidate_target_max_pearson': 'CTP', 
                 'candidate_target_max_spearman': 'CTS', 'candidate_target_max_covariance': 'CTC', 'candidate_target_max_mutual_info': 'CTM', 
                 'max_pearson_difference': 'PD', 'containment_fraction': 'CF'}

EVAL_COLUMN = 'eval_mean_based_class'
POSITIVE_CLASS = 'good_gain'
NEGATIVE_CLASS = 'loss'
CLASS_COLUMN = 'mean_based_class'

def remove_outliers_based_on_zscores(feature):
  mean_ = np.mean(feature)
  std_ = np.std(feature)
  return [i for i in feature if np.fabs((i - mean_)/std_) < OUTLIER_THRESHOLD_ZSCORES]

def remove_outliers_based_on_mad(feature):
  mad = median_absolute_deviation(feature)
  median = np.median(feature)
  return [i for i in feature if np.fabs((i - median)/mad) < OUTLIER_THRESHOLD_MAD]

def normalize_values(feature):
  try:
    feature /= np.max(np.fabs(feature),axis=0)
    return feature
  except ValueError:
    return []

def plot_features_and_target_histograms(data, prefix):
  for feature_name in FEATURES:
    #tmp = remove_outliers_based_on_zscores(data[feature_name])
    #tmp = remove_outliers_based_on_mad(tmp)
    tmp = remove_outliers_based_on_mad(data[feature_name])
    #tmp = data[feature_name]

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
  tmp = remove_outliers_based_on_mad(data[TARGET])

  weights = np.ones_like(tmp)/float(len(tmp))
  plt.hist(tmp, bins=50, alpha=0.5, weights=weights)
  plt.xlabel('Value Ranges')
  plt.ylabel('Percentages')
  plt.title(TARGET)
  plt.savefig(prefix + '-TARGET-' + TARGET + '.png',  dpi=600)
  plt.close()

def plot_two_kinds_of_histograms(data1, label1, data2, label2):
  """ This function plots histograms for features and targets, just like 
  function 'plot_features_and_target_histograms' for two different data (data1 and data2)
  """
  #print('max gain in data1:', max(data1['gain_in_r2_score']), 'max gain in data2:', max(data2['gain_in_r2_score']))
  for feature_name in ['containment_fraction']: #FEATURES:
    #tmp1 = normalize_values(remove_outliers_based_on_mad(data1[feature_name]))
    #tmp2 = normalize_values(remove_outliers_based_on_mad(data2[feature_name]))
    tmp1 = data1[feature_name] #remove_outliers_based_on_mad(data1[feature_name])
    tmp2 = data2[feature_name] #remove_outliers_based_on_mad(data2[feature_name])
    weights1 = np.ones_like(tmp1)/float(len(tmp1))
    weights2 = np.ones_like(tmp2)/float(len(tmp2))
    plt.hist(tmp1, bins=50, alpha=0.5, weights=weights1, label=label1, color='blue')
    plt.hist(tmp2, bins=50, alpha=0.5, weights=weights2, label=label2, color='red')
    plt.xlabel('Value Ranges')
    plt.ylabel('Percentages')
    plt.title('Query Containment Fraction') #ABBREVIATIONS[feature_name])
    plt.legend()
    plt.savefig(feature_name + '.png',  dpi=600)
    plt.close()

  tmp1 = data1[feature_name]
  tmp2 = data2[feature_name]
  weights1 = np.ones_like(tmp1)/float(len(tmp1))
  weights2 = np.ones_like(tmp2)/float(len(tmp2))
  plt.hist(tmp1, bins=50, alpha=0.5, weights=weights1, label=label1)
  plt.hist(tmp2, bins=50, alpha=0.5, weights=weights2, label=label2)
  plt.xlabel('Value Ranges')
  plt.ylabel('Percentages')
  plt.title(TARGET)
  plt.legend()
  plt.savefig(TARGET + '.png',  dpi=600)
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

def split_augmentations_into_positive_and_negative_class(data):
  """This function splits instances into those that correspond
  to positive class and negative class wrt a certain CLASS_COLUMN
  """
  positive = data.loc[data[CLASS_COLUMN] == POSITIVE_CLASS]
  negative = data.loc[data[CLASS_COLUMN] == NEGATIVE_CLASS]
  return positive, negative
  
if __name__=='__main__':
  use_case_dataset = pd.read_csv(sys.argv[1])    
  gain, loss = split_augmentations_into_gain_and_loss(use_case_dataset)
  plot_two_kinds_of_histograms(gain, 'successful', loss, 'unsuccessful')

  # plot_features_and_target_histograms(gain, 'successful')
  # plot_features_and_target_histograms(loss, 'unsuccessful')

  # positive, negative = split_augmentations_into_positive_and_negative_class(use_case_dataset)
  # plot_features_and_target_histograms(positive, 'positive class')
  # plot_features_and_target_histograms(negative, 'negative class')
