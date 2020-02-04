""" This script gets a file with the features in test-data-features-with-predictions-and-mse.header and does the following:

1 - plots scatterplots of features versus mean squared errors for when the latter are inside the interval [-1;1] but also when they are outside it
2 - plots scatterplots of features versus mean squared errors for different quadrants of the data
3 - plots scatterplots of real vs predicted gains for when the latter are inside the interval [-1;1] but also when they are outside it
4 - plots scatterplots of real vs predicted gains for different quadrants of the data 
5 - generates histogram for real gains in interval [-1;1]
6 - checks how often the predicted gains get the real sign right, plotting a histogram of mean squared errors for when signs get mixed 
7 - ranks query and candidate datasets based on the errors we get for them
"""

import sys
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

def plot_scatterplot(x, y, image_name, xlabel, ylabel):
  """Given aligned values x and y, this method generates a scatterplot of them.
  """
  plt.scatter(x, y, alpha=0.5)
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.tight_layout()
  plt.savefig(image_name, dpi=300)
  plt.close()


FIELDS = ['query_filename','target','candidate_filename','query_num_of_columns','query_num_of_rows','query_row_column_ratio','query_max_mean','query_max_outlier_percentage','query_max_skewness','query_max_kurtosis','query_max_unique','candidate_num_of_columns','candidate_num_rows','candidate_row_column_ratio','candidate_max_mean','candidate_max_outlier_percentage','candidate_max_skewness','candidate_max_kurtosis','candidate_max_unique','query_target_max_pearson','query_target_max_spearman','query_target_max_covariance','query_target_max_mutual_info','candidate_target_max_pearson','candidate_target_max_spearman','candidate_target_max_covariance','candidate_target_max_mutual_info','max_pearson_difference','containment_fraction','decrease_in_mae','decrease_in_mse','decrease_in_med_ae','gain_in_r2_score','predicted_gain_in_r2_score','mse_between_real_and_predicted_gains']

GAIN_IN_R2_INTERVAL = [-1, 1]

data = np.array([l.strip().split(',') for l in open(sys.argv[1]).readlines()])
mse_column_data = np.array([float(i) for i in data[:,-1]])
predicted_gain_data = np.array([float(i) for i in data[:,-2]])
real_gain_data = np.array([float(i) for i in data[:,-3]])

# for column_index in range(len(FIELDS) - 3):
#   print(column_index, len(FIELDS))
#   try:
#     column_data = np.array([float(i) for i in data[:,column_index]])
#   except ValueError:
#     print('the column contains either dataset or target names')
#     continue
#   if 'decrease_in' not in FIELDS[column_index]:
#     # cross feature with mse, predicted values and real values for all data
#     plot_scatterplot(column_data, mse_column_data,  FIELDS[column_index] + '_' + FIELDS[-1] + '.png', FIELDS[column_index], FIELDS[-1]) 
#     plot_scatterplot(column_data, predicted_gain_data,  FIELDS[column_index] + '_' + FIELDS[-2] + '.png', FIELDS[column_index], FIELDS[-2]) 
#     plot_scatterplot(column_data, real_gain_data,  FIELDS[column_index] + '_' + FIELDS[-3] + '.png', FIELDS[column_index], FIELDS[-3]) 

# data_gain_inside_interval = []
# data_gain_outside_interval = []
# for row in data:
#   real_gain = float(row[-3])
#   if real_gain >= GAIN_IN_R2_INTERVAL[0] and real_gain <= GAIN_IN_R2_INTERVAL[1]:
#     data_gain_inside_interval.append(row)
#   else:
#     data_gain_outside_interval.append(row) 
# data_gain_inside_interval = np.array(data_gain_inside_interval)
# data_gain_outside_interval = np.array(data_gain_outside_interval)
# inside_interval_mse_column_data = np.array([float(i) for i in data_gain_inside_interval[:,-1]])
# inside_interval_predicted_gain_data = np.array([float(i) for i in data_gain_inside_interval[:,-2]])
# inside_interval_real_gain_data = np.array([float(i) for i in data_gain_inside_interval[:,-3]])
# outside_interval_mse_column_data = np.array([float(i) for i in data_gain_outside_interval[:,-1]])
# outside_interval_predicted_gain_data = np.array([float(i) for i in data_gain_outside_interval[:,-2]])
# outside_interval_real_gain_data = np.array([float(i) for i in data_gain_outside_interval[:,-3]])
# for column_index in range(len(FIELDS) - 3):
#   print(column_index, len(FIELDS))
#   try:
#     inside_interval_column_data = np.array([float(i) for i in data_gain_inside_interval[:,column_index]])
#     outside_interval_column_data = np.array([float(i) for i in data_gain_outside_interval[:,column_index]])
#   except ValueError:
#     print('the column contains either dataset or target names')
#     continue
#   if 'decrease_in' not in FIELDS[column_index]:
#     # cross feature with mse, predicted values and real values for all data
#     plot_scatterplot(inside_interval_column_data, inside_interval_mse_column_data,  
#                      'inside_interval_' + FIELDS[column_index] + '_' + FIELDS[-1] + '.png', FIELDS[column_index], FIELDS[-1]) 
#     plot_scatterplot(inside_interval_column_data, inside_interval_predicted_gain_data,  
#                      'inside_interval_' + FIELDS[column_index] + '_' + FIELDS[-2] + '.png', FIELDS[column_index], FIELDS[-2]) 
#     plot_scatterplot(inside_interval_column_data, inside_interval_real_gain_data,  
#                      'inside_interval_' + FIELDS[column_index] + '_' + FIELDS[-3] + '.png', FIELDS[column_index], FIELDS[-3]) 
#     plot_scatterplot(outside_interval_column_data, outside_interval_mse_column_data,  
#                      'outside_interval_' + FIELDS[column_index] + '_' + FIELDS[-1] + '.png', FIELDS[column_index], FIELDS[-1]) 
#     plot_scatterplot(outside_interval_column_data, outside_interval_predicted_gain_data,  
#                      'outside_interval_' + FIELDS[column_index] + '_' + FIELDS[-2] + '.png', FIELDS[column_index], FIELDS[-2]) 
#     plot_scatterplot(outside_interval_column_data, outside_interval_real_gain_data,  
#                      'outside_interval_' + FIELDS[column_index] + '_' + FIELDS[-3] + '.png', FIELDS[column_index], FIELDS[-3]) 


quadrant_1_data = []
quadrant_2_data = []
quadrant_3_data = []
quadrant_4_data = []
for row in data:
  real_gain = float(row[-3])
  predicted_gain = float(row[-2])
  if real_gain > 0 and predicted_gain > 0:
    quadrant_1_data.append(row)
  elif real_gain <= 0 and predicted_gain > 0:
    quadrant_2_data.append(row)
  elif real_gain <= 0 and predicted_gain <= 0:
    quadrant_3_data.append(row)
  else:
    quadrant_4_data.append(row)

quadrant_1_data = np.array(quadrant_1_data)
quadrant_1_mse_column_data = np.array([float(i) for i in quadrant_1_data[:,-1]])
quadrant_1_predicted_gain_data = np.array([float(i) for i in quadrant_1_data[:,-2]])
quadrant_1_real_gain_data = np.array([float(i) for i in quadrant_1_data[:,-3]])

quadrant_2_data = np.array(quadrant_2_data)
quadrant_2_mse_column_data = np.array([float(i) for i in quadrant_2_data[:,-1]])
quadrant_2_predicted_gain_data = np.array([float(i) for i in quadrant_2_data[:,-2]])
quadrant_2_real_gain_data = np.array([float(i) for i in quadrant_2_data[:,-3]])

quadrant_3_data = np.array(quadrant_3_data)
quadrant_3_mse_column_data = np.array([float(i) for i in quadrant_3_data[:,-1]])
quadrant_3_predicted_gain_data = np.array([float(i) for i in quadrant_3_data[:,-2]])
quadrant_3_real_gain_data = np.array([float(i) for i in quadrant_3_data[:,-3]])

quadrant_4_data = np.array(quadrant_4_data)
quadrant_4_mse_column_data = np.array([float(i) for i in quadrant_4_data[:,-1]])
quadrant_4_predicted_gain_data = np.array([float(i) for i in quadrant_4_data[:,-2]])
quadrant_4_real_gain_data = np.array([float(i) for i in quadrant_4_data[:,-3]])

for column_index in range(len(FIELDS) - 3):
  print(column_index, len(FIELDS))
  try:
    quadrant_1_column_data = np.array([float(i) for i in quadrant_1_data[:,column_index]])
    quadrant_2_column_data = np.array([float(i) for i in quadrant_2_data[:,column_index]])
    quadrant_3_column_data = np.array([float(i) for i in quadrant_3_data[:,column_index]])
    quadrant_4_column_data = np.array([float(i) for i in quadrant_4_data[:,column_index]])
  except ValueError:
    print('the column contains either dataset or target names')
    continue
  if 'decrease_in' not in FIELDS[column_index]:
    # cross feature with mse, predicted values and real values for all data
    plot_scatterplot(quadrant_1_column_data, quadrant_1_mse_column_data,  
                     'quadrant_1_of_real_and_predicted_gains_' + FIELDS[column_index] + '_' + FIELDS[-1] + '.png', FIELDS[column_index], FIELDS[-1]) 
    plot_scatterplot(quadrant_1_column_data, quadrant_1_predicted_gain_data,  
                     'quadrant_1_of_real_and_predicted_gains_' + FIELDS[column_index] + '_' + FIELDS[-2] + '.png', FIELDS[column_index], FIELDS[-2]) 
    plot_scatterplot(quadrant_1_column_data, quadrant_1_real_gain_data,  
                     'quadrant_1_of_real_and_predicted_gains_' + FIELDS[column_index] + '_' + FIELDS[-3] + '.png', FIELDS[column_index], FIELDS[-3]) 
    plot_scatterplot(quadrant_2_column_data, quadrant_2_mse_column_data,  
                     'quadrant_2_of_real_and_predicted_gains_' + FIELDS[column_index] + '_' + FIELDS[-1] + '.png', FIELDS[column_index], FIELDS[-1]) 
    plot_scatterplot(quadrant_2_column_data, quadrant_2_predicted_gain_data,  
                     'quadrant_2_of_real_and_predicted_gains_' + FIELDS[column_index] + '_' + FIELDS[-2] + '.png', FIELDS[column_index], FIELDS[-2]) 
    plot_scatterplot(quadrant_2_column_data, quadrant_2_real_gain_data,  
                     'quadrant_2_of_real_and_predicted_gains_' + FIELDS[column_index] + '_' + FIELDS[-3] + '.png', FIELDS[column_index], FIELDS[-3]) 
    plot_scatterplot(quadrant_3_column_data, quadrant_3_mse_column_data,  
                     'quadrant_3_of_real_and_predicted_gains_' + FIELDS[column_index] + '_' + FIELDS[-1] + '.png', FIELDS[column_index], FIELDS[-1]) 
    plot_scatterplot(quadrant_3_column_data, quadrant_3_predicted_gain_data,  
                     'quadrant_3_of_real_and_predicted_gains_' + FIELDS[column_index] + '_' + FIELDS[-2] + '.png', FIELDS[column_index], FIELDS[-2]) 
    plot_scatterplot(quadrant_3_column_data, quadrant_3_real_gain_data,  
                     'quadrant_3_of_real_and_predicted_gains_' + FIELDS[column_index] + '_' + FIELDS[-3] + '.png', FIELDS[column_index], FIELDS[-3]) 
    plot_scatterplot(quadrant_4_column_data, quadrant_4_mse_column_data,  
                     'quadrant_4_of_real_and_predicted_gains_' + FIELDS[column_index] + '_' + FIELDS[-1] + '.png', FIELDS[column_index], FIELDS[-1]) 
    plot_scatterplot(quadrant_4_column_data, quadrant_4_predicted_gain_data,  
                     'quadrant_4_of_real_and_predicted_gains_' + FIELDS[column_index] + '_' + FIELDS[-2] + '.png', FIELDS[column_index], FIELDS[-2]) 
    plot_scatterplot(quadrant_4_column_data, quadrant_4_real_gain_data,  
                     'quadrant_4_of_real_and_predicted_gains_' + FIELDS[column_index] + '_' + FIELDS[-3] + '.png', FIELDS[column_index], FIELDS[-3]) 
