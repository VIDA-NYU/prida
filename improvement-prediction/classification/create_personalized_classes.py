""" Given 
      (1) a dataset file with FEATURES and a TARGET column,
 
    this script determines classes GOOD GAIN and LOSS on a 'personalized' basis. In practice, 
    for each query, we consider the 'gain scores' for all different augmentations (i.e., 
    for all different candidates), and determine which correspond to gains or losses 
    depending on the induced value distribution. 
"""

import pandas as pd
import sys
import numpy as np

QUERY_COLUMN = 'query'
CANDIDATE_COLUMN = 'candidate'
TARGET = 'gain_in_r2_score'
POSITIVE_CLASS = 'good_gain'
NEGATIVE_CLASS = 'loss'

def determine_class_for_query_augmentations(query_rows):
  """This function determines the class for each augmentation 
  involving a candidate c and the query q that induced parameter 
  query_rows
  """

  augmentation_classes = {}
  target_values = query_rows[TARGET]
  median = np.median(target_values)
  mean_ = np.mean(target_values)
  for index, row in query_rows.iterrows():
      med_tmp = row[TARGET] - median
      if med_tmp > 0 and row[TARGET] > 0:
          med_class = POSITIVE_CLASS
      else:
          med_class = NEGATIVE_CLASS

      mean_tmp = row[TARGET] - mean_
      if mean_tmp > 0 and row[TARGET] > 0:
          mean_class = POSITIVE_CLASS
      else:
          mean_class = NEGATIVE_CLASS
      
      key = row[QUERY_COLUMN] + '-' + row[CANDIDATE_COLUMN]
      augmentation_classes[key] = [med_class, mean_class]
  return augmentation_classes
  
def determine_class_based_on_target(data):
  """This function determines the class of each augmentation
  on a personalized basis.

  FIXME This function is a bit costly, executing two loops that seem a bit 
  redundant. Unfortunately, pandas did not accept proper inplace column 
  addition to different subparts of parameter data.
  """
  queries = set(data[QUERY_COLUMN])
  augmentations = {}
  for q in queries:
    tmp_augs = determine_class_for_query_augmentations(data.loc[data[QUERY_COLUMN] == q])
    augmentations.update(tmp_augs)

  mean_class = []
  median_class = []
  for index, row in data.iterrows():
      key = row[QUERY_COLUMN] + '-' + row[CANDIDATE_COLUMN]
      classes = augmentations[key]
      median_class.append(classes[0])
      mean_class.append(classes[1])
  data['median_based_class'] = median_class
  data['mean_based_class'] = mean_class
  return data

if __name__ == '__main__':

  dataset_filename = sys.argv[1]
  dataset = pd.read_csv(dataset_filename)
  data_with_classes = determine_class_based_on_target(dataset)
  f = open(dataset_filename.split('.')[0] + '_with_median_and_mean_based_classes.csv', 'w')
  f.write(data_with_classes.to_csv(index=False))
  f.close()
