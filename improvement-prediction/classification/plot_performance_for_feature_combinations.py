"""
This script gets a file with performances for different features and feature combinations in the following format:

[(feature_comb_1, recall_1), (feature_comb_1, precision_1), (feature_comb_1, fmeasure_1)]
.
.
.
[(feature_comb_m, recall_m), (feature_comb_m, precision_m), (feature_comb_m, fmeasure_m)]

And then generates a bar plot.
"""

import sys
import matplotlib.pyplot as plt
import numpy as np

ABBREV = {'query_max_kurtosis': 'q_kurtosis', 'query_max_skewness': 'q_skewness', 'query_row_column_ratio': 'q_row_to_col', 'query_num_of_columns': 'q_cols', 
          'query_max_unique': 'q_unique', 'query_num_of_rows': 'q_rows', 'query_target_max_mutual_info': 'q_mutinfo', 'query_target_max_covariance': 'q_covariance', 
          'query_target_max_pearson': 'q_pearson', 'query_target_max_spearman': 'q_spearman', 'candidate_max_kurtosis': 'c_kurtosis', 'candidate_max_unique': 'c_unique', 
          'candidate_num_rows': 'c_rows', 'candidate_row_column_ratio': 'c_row_to_col', 'candidate_max_skewness': 'c_skewness',  'candidate_num_of_columns': 'c_cols', 
          'candidate_target_max_mutual_info': 'c_mutinfo', 'candidate_target_max_spearman': 'c_spearman', 'candidate_target_max_pearson': 'c_pearson', 
          'candidate_target_max_covariance': 'c_covariance', 'containment_fraction': 'containment', 'dataset_feats': 'd_feats', 'dataset_target_feats': 'dt_feats', 
          'dataset_dataset_feats': 'dd_feats', 'dataset_feats_dataset_target_feats': 'd_dt_feats', 'dataset_feats_dataset_dataset_feats': 'd_dd_feats', 
          'dataset_dataset_feats_dataset_target_feats': 'dd_dt_feats', 'all_feats': 'all_feats'}

def plot_bars(features):
  labels = features.keys()

  recall = []; precision = []; fmeasure = []
  for key in features.keys():
    recall.append(features[key][0])
    precision.append(features[key][1])
    fmeasure.append(features[key][2])
  
  x = np.arange(len(labels))  # the label locations
  width = 0.2  # the width of the bars
  
  fig, ax = plt.subplots()
  rects0 = ax.bar(x - width/2, recall, width, alpha=0.6, color='#809a41', hatch='|', label='Recall')
  rects1 = ax.bar(x + width/2, precision, width, alpha=0.6, color='#1d3557', hatch='.', label='Precision')
  rects2 = ax.bar(x + 1.5*width, fmeasure, width, alpha=0.6, color='#e63946', label='F-measure')

# r f in features], 'o-', linewidth=3, label='recall')
#     axes.plot([abbrev[f[0]] for f in features], [f[1][1] for f in features], 'o--', linewidth=3, color='#1d3557', label='precision')
#     axes.plot([abbrev[f[0]] for f in features], [f[1][2] for f in features], 'o:', linewidth=3, color='#e63946', label='f-measure')
#     for label in axes.get_xti
#   rects2 = ax.bar(x + width/2, women_means, width, label='Women')
  
#   # Add some text for labels, title and custom x-axis tick labels, etc.
  ax.set_ylabel('Performance', fontsize=16)
#   ax.set_title('Scores by group and gender')
  ax.set_xticks(x)
  ax.set_xticklabels(labels)
  ax.set_xlabel('Feature Combinations', fontsize=16)
  ax.legend(loc='upper left', bbox_to_anchor=(0.6, 1.0), prop={'size':8})
  
#   def autolabel(rects):
#       """Attach a text label above each bar in *rects*, displaying its height."""
#       for rect in rects:
#           height = rect.get_height()
#           ax.annotate('{}'.format(height),
#                       xy=(rect.get_x() + rect.get_width() / 2, height),
#                       xytext=(0, 3),  # 3 points vertical offset
#                       textcoords="offset points",
#                       ha='center', va='bottom')
  
  
#   autolabel(rects1)
#   autolabel(rects2)
  
  fig.tight_layout()
  
  plt.savefig('tmp.png') #how()

lists = [eval(i.strip()) for i in open(sys.argv[1]).readlines() if '[(' in i]
features = {}
for list_ in lists:
    for feature in list_:
        if feature[0] in features:
            features[feature[0]].append(feature[1])
        else:
            features[feature[0]] = [feature[1]]

plot_bars(features)


#color_dict = {'query_max_kurtosis': 'lightblue', 'query_max_skewness': 'lightblue', 'query_row_column_ratio': 'lightblue', 'query_num_of_columns': 'lightblue', 
#               'query_max_unique': 'lightblue', 'query_num_of_rows': 'lightblue', 'query_target_max_mutual_info': 'blue', 'query_target_max_covariance': 'blue', 
#               'query_target_max_pearson': 'blue', 'query_target_max_spearman': 'blue', 'candidate_max_kurtosis': 'pink', 'candidate_max_unique': 'pink', 
#               'candidate_num_rows': 'pink', 'candidate_row_column_ratio': 'pink', 'candidate_max_skewness': 'pink', 'candidate_num_of_columns': 'pink', 
#               'candidate_max_skewness': 'pink', 'candidate_target_max_mutual_info': 'red', 'candidate_target_max_spearman': 'red', 
#               'candidate_target_max_pearson': 'red', 'candidate_target_max_covariance': 'red', 'containment_fraction': 'green', 'dataset_feats': 'brown', 
#               'dataset_target_feats': 'brown', 'dataset_dataset_feats': 'brown', 'dataset_feats_dataset_target_feats': 'brown', 'dataset_feats_dataset_dataset_feats': 'brown', 
#               'dataset_dataset_feats_dataset_target_feats': 'brown', 'all_feats': 'brown'}

#color_legend = {'lightblue': 'query', 'blue': 'query/target', 'pink': 'candidate', 'red': 'candidate/target', 'green': 'containment', 'combination': 'brown'}

def plot_boxplot(features_dict):
    features = sorted(features_dict.items(), key= lambda x: x[1], reverse=True)
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(18, 8))
    axes.plot([abbrev[f[0]] for f in features], [f[1][0] for f in features], 'o-', linewidth=3, color='#809a41', label='recall')
    axes.plot([abbrev[f[0]] for f in features], [f[1][1] for f in features], 'o--', linewidth=3, color='#1d3557', label='precision')
    axes.plot([abbrev[f[0]] for f in features], [f[1][2] for f in features], 'o:', linewidth=3, color='#e63946', label='f-measure')
    for label in axes.get_xticklabels():
        label.set_rotation(45)
        label.set_fontsize(26)
    axes.yaxis.grid(True, alpha=0.3)
    axes.yaxis.set_tick_params(labelsize=24)
    axes.set_xlabel('Feature Combinations', fontsize=32)
    axes.set_ylabel('Performance', fontsize=32)

#     q = mpatches.Patch(color='lightblue', label='query')
#     qt = mpatches.Patch(color='blue', label='query-target')
#     c = mpatches.Patch(color='pink', label='candidate')
#     ct = mpatches.Patch(color='red', label='candidate-target')
#     qc = mpatches.Patch(color='green', label='query-candidate')
#     cb = mpatches.Patch(color='brown', label='combination')
#     axes.legend(handles=[q, qt, c, ct, qc, cb], loc='upper right', prop={'size':22})
    axes.legend(loc='upper right', prop={'size':22})
    plt.savefig('feature_performance_plot.png', bbox_inches='tight', dpi=600)

