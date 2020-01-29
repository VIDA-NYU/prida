""" Given two files (one with training and one with validation data; both with headers), 
this script selects the threshold alpha above which a gain in R2 squared should correspond to 
class GOOD GAIN. Anything else will correspond to an instance of class LOSS.
"""

import pandas as pd
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report
from sklearn.utils import shuffle

ALPHA_GRID = [0.1 * x for x in range(10)]

def downsample_data(dataset):
  """This function downsamples the number of instances of a class that is over-represented in the dataset.
  It's important to keep the learning 'fair'
  """
  loss =  dataset.loc[dataset['classes'] == 'loss']
  good_gain = dataset.loc[dataset['classes'] == 'good_gain']
  
  sample_size = min([loss.shape[0], good_gain.shape[0]])
  loss = loss.sample(n=sample_size, random_state=42)
  good_gain = good_gain.sample(n=sample_size, random_state=42)
  
  frames = [loss, good_gain]
  return shuffle(pd.concat(frames), random_state=0)


def determine_classes_based_on_gain_in_r2_score(dataset, alpha, downsample=True):
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

  training = pd.read_csv(training_filename)
  validation = pd.read_csv(validation_filename)
  
  max_f1score = float('-inf')
  best_alpha = -1
  for alpha in ALPHA_GRID:
    tmp_training = determine_classes_based_on_gain_in_r2_score(training, alpha) #, downsample=False)
    tmp_validation = determine_classes_based_on_gain_in_r2_score(validation, alpha)#, downsample=False)

    X_train = tmp_training[['query_num_of_columns', 'query_num_of_rows', 'query_row_column_ratio',
       'query_max_mean', 'query_max_outlier_percentage', 'query_max_skewness',
       'query_max_kurtosis', 'query_max_unique', 'candidate_num_of_columns',
       'candidate_num_rows', 'candidate_row_column_ratio',
       'candidate_max_mean', 'candidate_max_outlier_percentage',
       'candidate_max_skewness', 'candidate_max_kurtosis',
       'candidate_max_unique', 'query_target_max_pearson',
       'query_target_max_spearman', 'query_target_max_covariance',
       'query_target_max_mutual_info', 'candidate_target_max_pearson',
       'candidate_target_max_spearman', 'candidate_target_max_covariance',
       'candidate_target_max_mutual_info', 'max_pearson_difference',
       'containment_fraction']]
    y_train = tmp_training['classes']

    X_test = tmp_validation[['query_num_of_columns', 'query_num_of_rows', 'query_row_column_ratio',
       'query_max_mean', 'query_max_outlier_percentage', 'query_max_skewness',
       'query_max_kurtosis', 'query_max_unique', 'candidate_num_of_columns',
       'candidate_num_rows', 'candidate_row_column_ratio',
       'candidate_max_mean', 'candidate_max_outlier_percentage',
       'candidate_max_skewness', 'candidate_max_kurtosis',
       'candidate_max_unique', 'query_target_max_pearson',
       'query_target_max_spearman', 'query_target_max_covariance',
       'query_target_max_mutual_info', 'candidate_target_max_pearson',
       'candidate_target_max_spearman', 'candidate_target_max_covariance',
       'candidate_target_max_mutual_info', 'max_pearson_difference',
       'containment_fraction']]
    y_test = tmp_validation['classes']

    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    f1score = f1_score(y_test, y_pred, pos_label='good_gain')
    if f1score > max_f1score:
      max_f1score = f1score
      best_alpha = alpha
    print('alpha', alpha, 'f1score', f1score)
    print(classification_report(y_test, y_pred))
  print('best alpha', best_alpha, 'f1-score', max_f1score)
