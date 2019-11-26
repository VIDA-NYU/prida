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

def plot_boxplot(features_dict):
    features = sorted(features_dict.items(), key= lambda x: sum(x[1]), reverse=True)
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(18, 8))
    bplot1 = axes.boxplot([f[1] for f in features],
                             vert=True,  # vertical box alignment
                             patch_artist=True,  # fill with color
                             labels=[f[0] for f in features])  # will be used to label x-ticks
    for label in axes.get_xticklabels():
        label.set_rotation(90)
    axes.set_title('Feature importances')

    # fill with colors
    colors = []
    # colors = ['pink', 'lightblue', 'lightgreen']
    # for bplot in (bplot1, bplot2):
    #     for patch, color in zip(bplot['boxes'], colors):
    #         patch.set_facecolor(color)

    #adding horizontal grid lines
    for ax in axes:
        ax.yaxis.grid(True)
        ax.set_xlabel('Features')
        ax.set_ylabel('Mean Decrease Impurity')
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
