""" Given 
      (1) a training model for a set of features FEATURES and a given class column X,
      (2) a test dataset file with FEATURES and the class column X,  and
      (3) the class column X
    this script generates classification reports for the test dataset (2) 
"""

import sys
import pandas as pd
import pickle
from sklearn.metrics import classification_report

FEATURES = ['query_num_of_columns', 'query_num_of_rows', 'query_row_column_ratio', 'query_max_skewness', 'query_max_kurtosis', 'query_max_unique', 'candidate_num_rows', 'candidate_row_column_ratio', 'candidate_max_skewness', 'candidate_max_kurtosis', 'candidate_max_unique', 'query_target_max_pearson', 'query_target_max_spearman', 'query_target_max_covariance', 'query_target_max_mutual_info', 'candidate_target_max_pearson', 'candidate_target_max_spearman', 'candidate_target_max_covariance', 'candidate_target_max_mutual_info', 'max_pearson_difference', 'containment_fraction']
BETA_1 = 0.6
BETA_2 = 0.4

def generate_classification_reports(model, test_data, class_column):
  """ This function generates classification reports taking all instances into account, 
  as well as only those that have prediction probability above BETA_1 or below BETA_2
  """
  preds = model.predict(test_data[FEATURES])
  preds_probs = model.predict_proba(test_data[FEATURES])
  print('**** classification of all instances')
  print(classification_report(test_data[class_column], preds))

  partial_tests = []
  partial_preds = []
  for pred, pred_prob, test in zip(preds, preds_probs, test_data[class_column]):
      if pred_prob[0] > BETA_1 or pred_prob[0] < BETA_2:
          partial_tests.append(test)
          partial_preds.append(pred)
  print('**** classification of instances with pred_proba above ', BETA_1, 'or below', BETA_2)
  print(classification_report(partial_tests, partial_preds))
  
  
if __name__=='__main__':
  model = pickle.load(open(sys.argv[1], 'rb'))
  test_data = pd.read_csv(sys.argv[2])
  class_column = sys.argv[3]
  generate_classification_reports(model, test_data, class_column)
