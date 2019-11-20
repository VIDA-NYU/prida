import matplotlib.pyplot as plt
import matplotlib.font_manager as font
import os
import sys


def parse_data(bins, y, initial_pos, stats_file):

    for i in range(initial_pos, initial_pos + 500):
        line = stats_file[i]

        # bin ranges
        bin_ranges = line[line.find('[')+1:line.find(']')].split(',')
        bins.append(float(bin_ranges[0]))
        if i == initial_pos + 499:  ## last one
            bins.append(float(bin_ranges[1]))

        # densities
        y.append(int(line.split('\t')[1]))


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
parse_data(
    n_rows_bins,
    n_rows_n,
    n_rows_position + 1,
    stats_file
)

# n. columns
n_columns_bins = list()
n_columns_n = list()
parse_data(
    n_columns_bins,
    n_columns_n,
    n_rows_position + 502,
    stats_file
)

# size in bytes
size_bytes_bins = list()
size_bytes_n = list()
parse_data(
    size_bytes_bins,
    size_bytes_n,
    n_rows_position + 1003,
    stats_file
)

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

make_histogram(
    size_bytes_bins,
    size_bytes_n,
    'Size (Bytes)',
    output_filename + '-size-bytes'
)
