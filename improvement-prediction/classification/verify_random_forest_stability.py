""" Given (1) two files (one with training and one with test data; both with headers), 
          (2) a threshold alpha above which a gain in R2 squared should correspond to class GOOD GAIN, and 
          (3) a file with the features that should be used for learning, 

          this script verifies the stability of results obtained with random forest classifiers with respect to different seeds.
"""

import sys
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from sklearn.metrics import classification_report
from scipy.stats import pearsonr, kendalltau
import numpy as np

TARGET_COLUMN = 'gain_in_r2_score'
POSITIVE_CLASS = 'good_gain'
NEGATIVE_CLASS = 'loss'

def determine_classes_based_on_gain_in_r2_score(dataset, alpha, downsample=False):
  """This function determines the class of each row in the dataset based on the value 
  of TARGET_COLUMN
  """
  gains = dataset[TARGET_COLUMN]
  classes = [POSITIVE_CLASS if i > alpha else NEGATIVE_CLASS for i in gains]
  dataset['class'] = classes
  if downsample:
    return downsample_data(dataset)
  return dataset

def generate_predictions_with_different_seeds(training, test, alpha, features):
  """This function creates different random forest classifiers with different seeds 
  and generates the corresponding predictions for the test data
  """
  training = determine_classes_based_on_gain_in_r2_score(training, alpha)
  test = determine_classes_based_on_gain_in_r2_score(test, alpha)
  X_train = training[features]
  y_train = training['class']
  X_test = test[features]
  y_test = test['class']

  clf = RandomForestClassifier(random_state=42, n_estimators=100)
  clf.fit(X_train, y_train)
  print(clf.feature_importances_)
  test['pred_42'] = clf.predict(X_test)
  test['prob_positive_class_42'] = [i[0] for i in clf.predict_proba(X_test)]
  clf = RandomForestClassifier(random_state=666, n_estimators=100)
  clf.fit(X_train, y_train)
  test['pred_666'] = clf.predict(X_test)
  test['prob_positive_class_666'] = [i[0] for i in clf.predict_proba(X_test)]

  # analyze flips in classes for different seeds
  try:
    flips = [row for index, row in test.iterrows() if row['pred_42'] != row['pred_666']]
    #print('flips', flips)
    f = open('test_records_that_flip.csv', 'w')
    f.write(pd.DataFrame(flips).to_csv(index=False))
    f.close()
  except ValueError:
      print('There were no flips, probably')

  try:
    non_flips = [row for index, row in test.iterrows() if row['pred_42'] == row['pred_666']]
    f = open('test_records_that_dont_flip.csv', 'w')
    f.write(pd.DataFrame(non_flips).to_csv(index=False))
    f.close()
  except ValueError:
      print('There were only flips, probably')
      
  return test
    
if __name__ == '__main__':
  training_filename = sys.argv[1]
  test_filename = sys.argv[2]
  alpha = float(sys.argv[3])
  features = eval(open(sys.argv[4]).readline())
  
  training_data = pd.read_csv(training_filename)
  test_data = pd.read_csv(test_filename)
  
  test_with_predictions = generate_predictions_with_different_seeds(training_data, test_data, alpha, features)
  
