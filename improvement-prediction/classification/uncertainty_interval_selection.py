""" Given two files (one with training and one with validation data; both with headers), this script performs 
classification, ignores predictions for which the prediction probability was within a certain interval BETA, and 
computes fmeasure using just the remaining predictions. The idea is to test different intervals BETA and pick the one 
that leads to the best fmeasure..
"""

INTERVALS = [(0.45, 0.55), (0.47, 0.53), (0.4, 0.6), (0.48, 0.52), (0.49, 0.51)]

FEATURES = ['query_num_of_columns', 'query_num_of_rows', 'query_row_column_ratio',  'query_max_skewness',
            'query_max_kurtosis', 'query_max_unique', 'candidate_num_of_columns', 'candidate_num_rows', 'candidate_row_column_ratio',
            'candidate_max_skewness', 'candidate_max_kurtosis',
            'candidate_max_unique', 'query_target_max_pearson','query_target_max_spearman', 'query_target_max_covariance',
            'query_target_max_mutual_info', 'candidate_target_max_pearson', 'candidate_target_max_spearman', 'candidate_target_max_covariance',
            'candidate_target_max_mutual_info', 'max_pearson_difference','containment_fraction']

import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score


def determine_classes_based_on_gain_in_r2_score(dataset):
  """This function determines the class of each row in the dataset based on the value 
  of column 'gain_in_r2_score'
  """
  gains = dataset['gain_in_r2_score']
  classes = ['good_gain' if i > 0 else 'loss' for i in gains]
  dataset['classes'] = classes
  return dataset

if __name__ == '__main__':
  
  training_filename = sys.argv[1]
  validation_filename = sys.argv[2]
  
  training = determine_classes_based_on_gain_in_r2_score(pd.read_csv(training_filename))
  validation = determine_classes_based_on_gain_in_r2_score(pd.read_csv(validation_filename))
  X_train = training[FEATURES]
  y_train = training['classes']
  
  X_test = validation[FEATURES]
  y_test = validation['classes']
  
  clf = RandomForestClassifier(random_state=42, n_estimators=100)
  clf.fit(X_train, y_train)
  y_pred = clf.predict(X_test)
  y_probs = clf.predict_proba(X_test)
  fscore = -1
  chosen_beta = None
  for beta in INTERVALS:
    tmp_test = []
    tmp_preds = []
    for real, pred, prob in zip(y_test, y_pred, y_probs):
      #print('prob', prob, 'beta', beta)
      if prob[0] < beta[0] or prob[0] > beta[1]:
        tmp_test.append(real)
        tmp_preds.append(pred)
    tmp_fscore = f1_score(tmp_test, tmp_preds, pos_label='good_gain')
    if tmp_fscore > fscore:
      fscore = tmp_fscore
      chosen_beta = beta
  print('chose beta', chosen_beta, 'with fscore', fscore)   
    
