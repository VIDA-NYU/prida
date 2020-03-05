""" Given (1) a dataset with features FEATURES and class column value for every instance, and  
          (2) the name of the class column

          this script generates TSNE two-dimensional plots to contrast the separation of instances wrt their classes
"""

import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# FEATURES = ['query_num_of_columns', 'query_num_of_rows', 'query_row_column_ratio', 'query_max_skewness', 'query_max_kurtosis', 'query_max_unique',
#             'candidate_num_of_columns', 'candidate_num_rows', 'candidate_row_column_ratio', 'candidate_max_skewness', 'candidate_max_kurtosis',
#             'candidate_max_unique', 'query_target_max_pearson', 'query_target_max_spearman', 'query_target_max_covariance','query_target_max_mutual_info',
#             'candidate_target_max_pearson', 'candidate_target_max_spearman', 'candidate_target_max_covariance', 'candidate_target_max_mutual_info',
#             'max_pearson_difference', 'containment_fraction']

FEATURES = ['query_row_column_ratio', 'candidate_target_max_pearson', 'candidate_target_max_spearman', 'max_pearson_difference', 'containment_fraction']
ALPHA = 0
TARGET_COLUMN = 'gain_in_r2_score'

def fit_tsne(data_filename, class_column='standard', use_sample=True, sample=2000):
  """This functions builds a classifier based on the training data.
  """
  
  training_data = pd.read_csv(data_filename).sample(n=sample) if use_sample else pd.read_csv(data_filename)
  if class_column == 'standard':
    training_data[class_column] = ['good_gain' if row[TARGET_COLUMN] > ALPHA else 'loss' for index, row in training_data.iterrows()]
  X_train = training_data[FEATURES]
  y_train = training_data[class_column]
  
  clf = TSNE(n_components=2,perplexity=100)
  y_pred = clf.fit_transform(X_train)
  return y_pred.T, y_train

if __name__ == '__main__':
  dataset_filename = sys.argv[1]
  class_column_name = sys.argv[2]

  y_pred_training, y_train_training = fit_tsne(dataset_filename, class_column_name, use_sample=False)
  colors = np.array(['green' if x=='good_gain' else 'red' for x in y_train_training])
  plt.scatter(y_pred_training[0], y_pred_training[1], c=colors, alpha=0.3)
  plt.savefig('tsne-plot-' + dataset_filename.split('.')[0], dpi=600)
