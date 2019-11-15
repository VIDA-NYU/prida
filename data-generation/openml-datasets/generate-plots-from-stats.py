import matplotlib.pyplot as plt
import matplotlib.font_manager as font
import os
import sys


def make_histogram(bins, values, x_axis, filename):

    f, ax = plt.subplots()
    ax.set_facecolor('#E0E0E0')

    plt.bar(
        height=values,
        x=bins[:-1],
        width=[bins[i]-bins[i-1] for i in range(1,len(bins))],
        color='#0099cc',
        edgecolor='#000000',
        align='edge'
    )

    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    ax.set_xlabel(x_axis)
    ax.set_xticks(bins)
    ax.set_xticklabels(['%.2f'%i for i in bins])

    ax.grid(b=True, axis='both', color='w', linestyle='-', linewidth=0.7)
    ax.set_axisbelow(True)

    f.set_size_inches(12, 10)

    plt.savefig(filename + '.png', bbox_inches='tight', pad_inches=0.05)
    plt.clf()


# stats file
stats_filename = sys.argv[1]
stats_file = open(stats_filename).read().split('\n')

n_rows_position = stats_file.index(' -- N. Rows:')

# filename
output_filename = sys.argv[2]

# n. rows
n_rows_bins = list()
n_rows_n = list()
for i in range(n_rows_position + 1, n_rows_position + 11):
    line = stats_file[i]

    # bin ranges
    bin_ranges = line[line.find('[')+1:line.find(']')].split(',')
    n_rows_bins.append(float(bin_ranges[0]))
    if i == n_rows_position + 10:  ## last one
        n_rows_bins.append(float(bin_ranges[1]))

    # densities
    n_rows_n.append(int(line.split('\t')[1]))

# n. columns
n_columns_bins = list()
n_columns_n = list()
for i in range(n_rows_position + 12, n_rows_position + 22):
    line = stats_file[i]

    # bin ranges
    bin_ranges = line[line.find('[')+1:line.find(']')].split(',')
    n_columns_bins.append(float(bin_ranges[0]))
    if i == n_rows_position + 21:  ## last one
        n_columns_bins.append(float(bin_ranges[1]))

    # densities
    n_columns_n.append(int(line.split('\t')[1]))

# plots

plt.figure(figsize=(12, 10), dpi=80)

make_histogram(
    n_rows_bins,
    n_rows_n,
    'Number of Rows',
    output_filename + '-n-rows'
)

make_histogram(
    n_columns_bins,
    n_columns_n,
    'Number of Columns',
    output_filename + '-n-columns'
)
