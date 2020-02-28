""" Given (1) a use case dataset with features FEATURES and target TARGET, 
          (2) a classifier model, compatible with library pickle, trained on FEATURES to predict a class derived from TARGET

          this script derives a copy of the use case dataset with added column 'eval', indicating whether each instance is a 
          false positive (FP), true positive (TP), false negative (FN), or true negative (TN) according to the classifier
"""

import sys
import pandas as pd
import pickle


TARGET = 'gain_in_r2_score'
FEATURES = ['query_num_of_columns', 'query_num_of_rows', 'query_row_column_ratio', 'query_max_skewness', 'query_max_kurtosis',
            'query_max_unique', 'candidate_num_of_columns', 'candidate_num_rows', 'candidate_row_column_ratio', 'candidate_max_skewness',
            'candidate_max_kurtosis', 'candidate_max_unique', 'query_target_max_pearson', 'query_target_max_spearman', 'query_target_max_covariance',
            'query_target_max_mutual_info', 'candidate_target_max_pearson', 'candidate_target_max_spearman', 'candidate_target_max_covariance',
            'candidate_target_max_mutual_info', 'max_pearson_difference', 'containment_fraction']
ALPHA = 0
POSITIVE_CLASS = 'good_gain'
NEGATIVE_CLASS = 'loss'

def determine_classes_based_on_target(dataset):
  """This function determines the class of each row in the dataset based on the value 
  of TARGET_COLUMN
  """
  gains = dataset[TARGET]
  dataset['class'] = [POSITIVE_CLASS if i > ALPHA else NEGATIVE_CLASS for i in gains]
  return dataset

def generate_predictions(classifier, test_data):
  """This function generates predictions for the class of each 
  test_data instance, and then checks whether they correspond to 
  FP, TP, FN, or TN
  """
  test_data = determine_classes_based_on_target(test_data)
  X_test = test_data[FEATURES]
  y_test = test_data['class']
  y_pred = classifier.predict(X_test)
  eval = []
  for t, p in zip(y_test, y_pred):
      if t == POSITIVE_CLASS and p == POSITIVE_CLASS:
          eval.append('tp')
      elif t == POSITIVE_CLASS and p == NEGATIVE_CLASS:
          eval.append('fn')
      elif t == NEGATIVE_CLASS and p == POSITIVE_CLASS:
          eval.append('fp')
      else:
          eval.append('tn')
  test_data['p(gain)'] = [i[0] for i in classifier.predict_proba(X_test)]
  test_data['pred_class'] = classifier.predict(X_test)
  test_data['eval'] = eval
  return test_data

if __name__ == '__main__':

    use_case_filename = sys.argv[1]
    model_filename = sys.argv[2]
    classifier_model = pickle.load(open(model_filename, 'rb'))

    use_case_data = pd.read_csv(use_case_filename)
    test_data_with_eval = generate_predictions(classifier_model, use_case_data)
    f = open(use_case_filename.split('.')[0] + '-MODEL-' + model_filename.split('.')[0] + '.csv' , 'w')
    f.write(test_data_with_eval.to_csv(index=False))
    f.close()
