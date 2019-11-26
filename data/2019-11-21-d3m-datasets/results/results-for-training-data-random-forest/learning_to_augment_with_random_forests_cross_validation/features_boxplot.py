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

color_dict = {'query_max_kurtosis': 'lightblue', 'query_max_mean': 'lightblue', 'query_row_column_ratio': 'lightblue', 'query_num_of_columns': 'lightblue', 'query_max_outlier_percentage': 'lightblue', 'query_max_unique': 'lightblue', 'query_num_of_rows': 'lightblue', 'query_target_max_mutual_info': 'blue', 'query_target_max_covariance': 'blue', 'query_target_max_pearson': 'blue', 'query_target_max_spearman': 'blue', 'candidate_max_kurtosis': 'pink', 'candidate_max_unique': 'pink', 'candidate_num_of_rows': 'pink', 'candidate_max_mean': 'pink', 'candidate_row_column_ratio': 'pink', 'candidate_max_skewness': 'pink', 'candidate_max_outlier_percentage': 'pink', 'candidate_num_of_columns': 'pink', 'candidate_max_skewness': 'pink', 'candidate_target_max_mutual_info': 'red', 'candidate_target_max_spearman': 'red', 'candidate_target_max_pearson': 'red', 'candidate_target_max_covariance': 'red', 'containment_fraction': 'green', 'max_pearson_difference': 'yellow'}

color_legend = {'lightblue': 'query', 'blue': 'query/target', 'pink': 'candidate', 'red': 'candidate/target', 'green': 'containment', 'yellow': 'pearson diff'}

def plot_boxplot(features_dict):
    features = sorted(features_dict.items(), key= lambda x: sum(x[1]), reverse=True)
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(18, 8))
    bplot = axes.boxplot([f[1] for f in features],
                             vert=True,  # vertical box alignment
                             patch_artist=True,  # fill with color
                             labels=[f[0] for f in features])  # will be used to label x-ticks
    for label in axes.get_xticklabels():
        label.set_rotation(90)
    axes.set_title('Feature importances')

    # fill with colors
    colors = [color_dict[f[0]] for f in features]
    indices_for_caption = {}
    i = 0
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_label(color)
        if color not in indices_for_caption:
            indices_for_caption[color_legend[color]] = bplot["boxes"][i]
        i += 1
    #adding horizontal grid lines
    axes.yaxis.grid(True)
    axes.set_xlabel('Features')
    axes.set_ylabel('Mean Decrease Impurity')
    axes.legend([i for i in indices_for_caption.values()], [i for i in indices_for_caption.keys()], loc='upper right')
    plt.savefig('feature-boxplot.png', bbox_inches='tight')

lists = [eval(i.strip()) for i in open(sys.argv[1]).readlines() if '[(' in i]
features = {}
for list_ in lists:
    for feature in list_:
        if feature[0] in features:
            features[feature[0]].append(feature[1])
        else:
            features[feature[0]] = [feature[1]]

plot_boxplot(features)
