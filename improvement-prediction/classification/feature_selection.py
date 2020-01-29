""" Given two files (one with training and one with validation data; both with headers) and an 
alpha parameter that splits instances into two classes (good_gain and loss), this script performs 
feature selection exploring a few different techniques.
"""

import sys
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report
from sklearn.feature_selection import RFE

VARIANCE_THRESHOLD = 0.01
CORRELATION_THRESHOLD = 0.85
FEATURES = ['query_num_of_columns', 'query_num_of_rows', 'query_row_column_ratio', 'query_max_mean', 'query_max_outlier_percentage', 'query_max_skewness',
            'query_max_kurtosis', 'query_max_unique', 'candidate_num_of_columns', 'candidate_num_rows', 'candidate_row_column_ratio',
            'candidate_max_mean', 'candidate_max_outlier_percentage','candidate_max_skewness', 'candidate_max_kurtosis',
            'candidate_max_unique', 'query_target_max_pearson','query_target_max_spearman', 'query_target_max_covariance',
            'query_target_max_mutual_info', 'candidate_target_max_pearson', 'candidate_target_max_spearman', 'candidate_target_max_covariance',
            'candidate_target_max_mutual_info', 'max_pearson_difference','containment_fraction']

def remove_low_variance_features(train_features, test_features):
  """This function removes features that have low variance in train_features
  """
  low_variance_filter = VarianceThreshold(threshold=VARIANCE_THRESHOLD)
  low_variance_filter.fit(train_features)
  low_variance_columns = [column for column in train_features.columns if column not in train_features.columns[low_variance_filter.get_support()]]
  print('low variance columns', low_variance_columns)
  return low_variance_filter.transform(train_features), low_variance_filter.transform(test_features)

def remove_highly_correlated_features(train_features, test_features):
  """This function removes features X that correlate too much with other features Y 
  (|Pearson coeff| > CORRELATION_THRESHOLD) in train_features
  """
  correlated_features = set()
  correlation_matrix = pd.DataFrame(train_features).corr()
  for i in range(len(correlation_matrix.columns)):
    for j in range(i):
      if abs(correlation_matrix.iloc[i, j]) > CORRELATION_THRESHOLD:
        colname = correlation_matrix.columns[i]
        correlated_features.add(colname)
  train_features = pd.DataFrame(train_features).drop(labels=correlated_features, axis=1)
  test_features = pd.DataFrame(test_features).drop(labels=correlated_features, axis=1)
  return train_features, test_features

def determine_classes_based_on_gain_in_r2_score(dataset, alpha, downsample=False):
  """This function determines the class of each row in the dataset based on the value 
  of column 'gain_in_r2_score'
  """
  gains = dataset['gain_in_r2_score']
  classes = ['good_gain' if i > alpha else 'loss' for i in gains]
  dataset['classes'] = classes
  if downsample:
    return downsample_data(dataset)
  return dataset

if __name__ == '__main__':
  
  training_filename = sys.argv[1]
  validation_filename = sys.argv[2]
  alpha = float(sys.argv[3])

  training = determine_classes_based_on_gain_in_r2_score(pd.read_csv(training_filename), alpha)
  validation = determine_classes_based_on_gain_in_r2_score(pd.read_csv(validation_filename), alpha)
  X_train = training[FEATURES]
  y_train = training['classes']
  
  X_test = validation[FEATURES]
  y_test = validation['classes']
  

  print('before', pd.DataFrame(X_train).columns)

#   clf = RandomForestClassifier(random_state=42)
#   num_features = len(FEATURES)
#   fscores = []
#   features = []
#   while num_features > 10:
#     rfe = RFE(estimator=clf, n_features_to_select=num_features, step=1)
#     rfe.fit(X_train, y_train)
#     tmp_X_train = rfe.transform(X_train)
#     tmp_X_test = rfe.transform(X_test)
#     clf.fit(tmp_X_train, y_train)
#     y_pred = clf.predict(tmp_X_test)
#     fscores.append(f1_score(y_test, y_pred, pos_label='good_gain'))
#     features.append([f for f, r in zip(FEATURES, list(rfe.ranking_)) if r == 1])
#     num_features -= 1
  
#   for f, feat in zip(fscores, features):
#     print(f, feat)

  clf = RandomForestClassifier(random_state=42)
  clf.fit(X_train, y_train)
  y_pred = clf.predict(X_test)
  print(classification_report(y_test, y_pred))
  
  X_train, X_test = remove_low_variance_features(X_train, X_test)
  print('after removing low variance', pd.DataFrame(X_train).columns)
  clf.fit(X_train, y_train)
  y_pred = clf.predict(X_test)
  print(classification_report(y_test, y_pred))


  X_train, X_test = remove_highly_correlated_features(X_train, X_test)
  print('after removing highly correlated features', pd.DataFrame(X_train).columns)
  clf.fit(X_train, y_train)
  y_pred = clf.predict(X_test)
  print(classification_report(y_test, y_pred))
