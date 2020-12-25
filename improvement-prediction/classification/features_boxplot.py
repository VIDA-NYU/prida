"""
This script gets a file with feature importance lists in the following format:

[(feature_name_11, feature_importance_11), , ..., (feature_name_1n, feature_importance_1n)]
.
.
.
[(feature_name_m1, feature_importance_m1), , ..., (feature_name_mn, feature_importance_mn)]

And then generate a box plot ranking these features.
"""

import sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

abbrev = {'query_max_mean': 'q_mean',          
          'query_max_kurtosis': 'q_kurtosis', 
          'query_max_skewness': 'q_skewness', 
          'query_row_column_ratio': 'q_row_to_col', 
          'query_num_of_columns': 'q_cols',
          'query_max_unique': 'q_unique', 
          'query_num_of_rows': 'q_rows', 
          'query_max_outlier_percentage': 'q_out_perc',
          'query_target_max_mutual_info': 'q_mutinfo', 
          'query_target_max_covariance': 'q_covariance',
          'query_target_max_pearson': 'q_pearson', 
          'query_target_max_spearman': 'q_spearman', 
          'candidate_max_kurtosis': 'c_kurtosis', 
          'candidate_max_unique': 'c_unique', 
          'candidate_num_rows': 'c_rows', 
          'candidate_max_mean': 'c_mean', 
          'candidate_max_outlier_percentage': 'c_out_perc', 
          'candidate_row_column_ratio': 'c_row_to_col', 
          'candidate_max_skewness': 'c_skewness', 
          'candidate_num_of_columns': 'c_cols', 
          'candidate_target_max_mutual_info': 'c_mutinfo', 
          'candidate_target_max_spearman': 'c_spearman', 
          'candidate_target_max_pearson': 'c_pearson', 
          'candidate_target_max_covariance': 'c_covariance', 
          'containment_fraction': 'containment'}

color_dict = {'query_max_mean': 'lightblue',
              'query_max_outlier_percentage': 'lightblue',
              'query_max_kurtosis': 'lightblue', 
              'query_max_skewness': 'lightblue', 
              'query_row_column_ratio': 'lightblue', 
              'query_num_of_columns': 'lightblue',
              'query_max_unique': 'lightblue', 
              'query_num_of_rows': 'lightblue', 
              'query_target_max_mutual_info': 'blue', 
              'query_target_max_covariance': 'blue',
              'query_target_max_pearson': 'blue', 
              'query_target_max_spearman': 'blue', 
              'candidate_max_kurtosis': 'pink', 
              'candidate_max_unique': 'pink', 
              'candidate_max_mean': 'pink',
              'candidate_max_outlier_percentage': 'pink',
              'candidate_num_rows': 'pink', 
              'candidate_row_column_ratio': 'pink',
              'candidate_max_skewness': 'pink', 
              'candidate_num_of_columns': 'pink', 
              'candidate_max_skewness': 'pink', 
              'candidate_target_max_mutual_info': 'red', 
              'candidate_target_max_spearman': 'red', 
              'candidate_target_max_pearson': 'red', 
              'candidate_target_max_covariance': 'red', 
              'containment_fraction': 'green'}

color_legend = {'lightblue': 'query', 'blue': 'query/target', 'pink': 'candidate', 'red': 'candidate/target', 'green': 'containment'}

def plot_boxplot(features_dict):
    features = sorted(features_dict.items(), key= lambda x: x[1], reverse=True)
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(18, 8))
    bplot = axes.bar([abbrev[f[0]] for f in features], [f[1][0] for f in features],
                      width = 0.5,
                      color = [color_dict[f[0]] for f in features])


 #[f[1] for f in features],
                             #vert=True,  # vertical box alignment
                             #patch_artist=True,  # fill with color
                             #labels=[abbrev[f[0]] for f in features])  # will be used to label x-ticks
    for label in axes.get_xticklabels():
        label.set_rotation(90)
        label.set_fontsize(28)
    #axes.set_title('Feature importances')

#     # fill with colors
#     colors = [color_dict[f[0]] for f in features]
#     indices_for_caption = {}
#     i = 0
#     for patch, color in zip(bplot['boxes'], colors):
#         patch.set_facecolor(color)
#         patch.set_label(color)
#         if color not in indices_for_caption:
#             indices_for_caption[color_legend[color]] = bplot["boxes"][i]
#         i += 1
    #adding horizontal grid lines
    axes.yaxis.grid(True)
    axes.yaxis.set_tick_params(labelsize=24)
    axes.set_xlabel('Features', fontsize=32) #, fontweight='bold')
    axes.set_ylabel('Gini Importance', fontsize=32) #, fontweight='bold')

#color_legend = {'lightblue': 'query', 'blue': 'query/target', 'pink': 'candidate', 'red': 'candidate/target', 'green': 'containment'}
    q = mpatches.Patch(color='lightblue', label='query')
    qt = mpatches.Patch(color='blue', label='query-target')
    c = mpatches.Patch(color='pink', label='candidate')
    ct = mpatches.Patch(color='red', label='candidate-target')
    #qc = mpatches.Patch(color='green', label='query-candidate')
    axes.legend(handles=[q, qt, c, ct],
                loc='upper right', prop={'size':22})

    #axes.legend([i for i in indices_for_caption.values()], [i for i in indices_for_caption.keys()], loc='upper right', prop={'size':22})
    plt.savefig('feature-boxplot.png', dpi=600, bbox_inches='tight')

lists = [eval(i.strip()) for i in open(sys.argv[1]).readlines() if '[(' in i]
features = {}
for list_ in lists:
    for feature in list_:
        if feature[0] in features:
            features[feature[0]].append(feature[1])
        else:
            features[feature[0]] = [feature[1]]

plot_boxplot(features)
