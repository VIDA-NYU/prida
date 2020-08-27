"""
This script gets a file with performances for different features and feature combinations in the following format:

[(feature_comb_11, performance_11), , ..., (feature_comb_1n, performance_1n)]
.
.
.
[(feature_comb_m1, performance_m1), , ..., (feature_comb_mn, performance_mn)]

And then generates a bar plot ranking these performances.
"""

import sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

abbrev = {'query_max_kurtosis': 'q_kurtosis', 'query_max_skewness': 'q_skewness', 'query_row_column_ratio': 'q_row_to_col', 'query_num_of_columns': 'q_cols', 
          'query_max_unique': 'q_unique', 'query_num_of_rows': 'q_rows', 'query_target_max_mutual_info': 'q_mutinfo', 'query_target_max_covariance': 'q_covariance', 
          'query_target_max_pearson': 'q_pearson', 'query_target_max_spearman': 'q_spearman', 'candidate_max_kurtosis': 'c_kurtosis', 'candidate_max_unique': 'c_unique', 
          'candidate_num_rows': 'c_rows', 'candidate_row_column_ratio': 'c_row_to_col', 'candidate_max_skewness': 'c_skewness',  'candidate_num_of_columns': 'c_cols', 
          'candidate_target_max_mutual_info': 'c_mutinfo', 'candidate_target_max_spearman': 'c_spearman', 'candidate_target_max_pearson': 'c_pearson', 
          'candidate_target_max_covariance': 'c_covariance', 'containment_fraction': 'containment', 'dataset_feats': 'd_feats', 'dataset_target_feats': 'dt_feats', 
          'dataset_dataset_feats': 'dd_feats', 'dataset_feats_dataset_target_feats': 'd_dt_feats', 'dataset_feats_dataset_dataset_feats': 'd_dd_feats', 
          'dataset_dataset_feats_dataset_target_feats': 'dd_dt_feats', 'all_feats': 'all_feats'}




color_dict = {'query_max_kurtosis': 'lightblue', 'query_max_skewness': 'lightblue', 'query_row_column_ratio': 'lightblue', 'query_num_of_columns': 'lightblue', 
              'query_max_unique': 'lightblue', 'query_num_of_rows': 'lightblue', 'query_target_max_mutual_info': 'blue', 'query_target_max_covariance': 'blue', 
              'query_target_max_pearson': 'blue', 'query_target_max_spearman': 'blue', 'candidate_max_kurtosis': 'pink', 'candidate_max_unique': 'pink', 
              'candidate_num_rows': 'pink', 'candidate_row_column_ratio': 'pink', 'candidate_max_skewness': 'pink', 'candidate_num_of_columns': 'pink', 
              'candidate_max_skewness': 'pink', 'candidate_target_max_mutual_info': 'red', 'candidate_target_max_spearman': 'red', 
              'candidate_target_max_pearson': 'red', 'candidate_target_max_covariance': 'red', 'containment_fraction': 'green', 'dataset_feats': 'brown', 
              'dataset_target_feats': 'brown', 'dataset_dataset_feats': 'brown', 'dataset_feats_dataset_target_feats': 'brown', 'dataset_feats_dataset_dataset_feats': 'brown', 
              'dataset_dataset_feats_dataset_target_feats': 'brown', 'all_feats': 'brown'}

color_legend = {'lightblue': 'query', 'blue': 'query/target', 'pink': 'candidate', 'red': 'candidate/target', 'green': 'containment', 'combination': 'brown'}

def plot_boxplot(features_dict):
    features = sorted(features_dict.items(), key= lambda x: x[1], reverse=True)
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(18, 8))
    axes.plot([abbrev[f[0]] for f in features], [f[1][0] for f in features], 'o-', linewidth=3, color='#809a41', label='recall')
    axes.plot([abbrev[f[0]] for f in features], [f[1][1] for f in features], 'o--', linewidth=3, color='#1d3557', label='precision')
    axes.plot([abbrev[f[0]] for f in features], [f[1][2] for f in features], 'o:', linewidth=3, color='#e63946', label='f-measure')
    for label in axes.get_xticklabels():
        label.set_rotation(45)
        label.set_fontsize(24)
    axes.yaxis.grid(True, alpha=0.3)
    axes.yaxis.set_tick_params(labelsize=24)
    axes.set_xlabel('Feature Combinations', fontsize=28)
    axes.set_ylabel('Performance', fontsize=28)

#     q = mpatches.Patch(color='lightblue', label='query')
#     qt = mpatches.Patch(color='blue', label='query-target')
#     c = mpatches.Patch(color='pink', label='candidate')
#     ct = mpatches.Patch(color='red', label='candidate-target')
#     qc = mpatches.Patch(color='green', label='query-candidate')
#     cb = mpatches.Patch(color='brown', label='combination')
#     axes.legend(handles=[q, qt, c, ct, qc, cb], loc='upper right', prop={'size':22})
    axes.legend(loc='upper right', prop={'size':22})
    plt.savefig('feature_performance_plot.png', bbox_inches='tight', dpi=600)

lists = [eval(i.strip()) for i in open(sys.argv[1]).readlines() if '[(' in i]
features = {}
for list_ in lists:
    for feature in list_:
        if feature[0] in features:
            features[feature[0]].append(feature[1])
        else:
            features[feature[0]] = [feature[1]]

plot_boxplot(features)
