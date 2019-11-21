import matplotlib.pyplot as plt
import matplotlib.font_manager as font
import os
import sys


def parse_data(bins, y, initial_pos, stats_file):

    for i in range(initial_pos, initial_pos + 10):
        line = stats_file[i]

        # bin ranges
        bin_ranges = line[line.find('[')+1:line.find(']')].split(',')
        bins.append(float(bin_ranges[0]))
        if i == initial_pos + 9:  ## last one
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

    plt.savefig('plots/' + filename + '.png', bbox_inches='tight', pad_inches=0.05)
    plt.clf()


def autolabel(ax, bar, data):
    rects = bar.patches
    position_x = list()
    position_y = list()
    for rect in rects:
        position_x.append(rect.get_x()+rect.get_width()/2.)
        position_y.append(rect.get_height())
    # attach some text labels
    for i in range(len(data)):
        ax.text(position_x[i], position_y[i] + 0.5,
                data[i], ha='center', va='bottom',
                fontproperties=font.FontProperties(style='italic',weight='bold'))


def make_before_after_histogram(values, filename):

    bin_width = 0.5

    scores = ['MAE', 'MSE', 'MAD', 'R^2']
    bins = list()
    x_ticks = list()
    initial_bin = bin_width
    for _ in scores:
        bins.append(initial_bin)
        bins.append(initial_bin + bin_width)
        x_ticks.append(initial_bin + bin_width)
        initial_bin += 3*bin_width

    f, ax = plt.subplots()
    ax.set_facecolor('#E0E0E0')

    bar = plt.bar(
        height=values,
        x=bins,
        width=bin_width,
        color='#0099cc',
        edgecolor='#000000',
        align='edge'
    )
    autolabel(ax, bar, ['b > a', 'b < a']*len(scores))

    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    ax.set_ylabel('N. Training Records (%)')

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(scores)

    ax.grid(b=True, axis='both', color='w', linestyle='-', linewidth=0.7)
    ax.set_axisbelow(True)

    f.set_size_inches(12, 10)

    plt.savefig('plots/' + filename + '.png', bbox_inches='tight', pad_inches=0.05)
    plt.clf()


# stats file
stats_filename = sys.argv[1]
stats_file = open(stats_filename).read().split('\n')

general_stats_position = stats_file.index('General statistics:')

# filename
output_filename = sys.argv[2]

# query intersection size
query_intersection_size_bins = list()
query_intersection_size_n = list()
parse_data(
    query_intersection_size_bins,
    query_intersection_size_n,
    general_stats_position + 4,
    stats_file
)

# candidate intersection size
candidate_intersection_size_bins = list()
candidate_intersection_size_n = list()
parse_data(
    candidate_intersection_size_bins,
    candidate_intersection_size_n,
    general_stats_position + 15,
    stats_file
)

# query number of records
query_n_records_bins = list()
query_n_records_n = list()
parse_data(
    query_n_records_bins,
    query_n_records_n,
    general_stats_position + 26,
    stats_file
)

# query number of columns
query_n_columns_bins = list()
query_n_columns_n = list()
parse_data(
    query_n_columns_bins,
    query_n_columns_n,
    general_stats_position + 37,
    stats_file
)

# candidate number of records
candidate_n_records_bins = list()
candidate_n_records_n = list()
parse_data(
    candidate_n_records_bins,
    candidate_n_records_n,
    general_stats_position + 48,
    stats_file
)

# candidate number of columns
candidate_n_columns_bins = list()
candidate_n_columns_n = list()
parse_data(
    candidate_n_columns_bins,
    candidate_n_columns_n,
    general_stats_position + 59,
    stats_file
)

# join size
join_size_bins = list()
join_size_n = list()
parse_data(
    join_size_bins,
    join_size_n,
    general_stats_position + 70,
    stats_file
)

# query size in bytes
query_size_bytes_bins = list()
query_size_bytes_n = list()
parse_data(
    query_size_bytes_bins,
    query_size_bytes_n,
    general_stats_position + 81,
    stats_file
)

# candidate size in bytes
candidate_size_bytes_bins = list()
candidate_size_bytes_n = list()
parse_data(
    candidate_size_bytes_bins,
    candidate_size_bytes_n,
    general_stats_position + 92,
    stats_file
)

# plots

if not os.path.exists('plots'):
    os.mkdir('plots')

plt.figure(figsize=(12, 10), dpi=80)

make_histogram(
    query_intersection_size_bins,
    query_intersection_size_n,
    'Join Intersection Size (Query Dataset)',
    output_filename + '-query-intersection-size'
)

make_histogram(
    candidate_intersection_size_bins,
    candidate_intersection_size_n,
    'Join Intersection Size (Candidate Dataset)',
    output_filename + '-candidate-intersection-size'
)

make_histogram(
    query_n_records_bins,
    query_n_records_n,
    'Number of Records (Query Dataset)',
    output_filename + '-query-n-records'
)

make_histogram(
    query_n_columns_bins,
    query_n_columns_n,
    'Number of Columns (Query Dataset)',
    output_filename + '-query-n-columns'
)

make_histogram(
    candidate_n_records_bins,
    candidate_n_records_n,
    'Number of Records (Candidate Dataset)',
    output_filename + '-candidate-n-records'
)

make_histogram(
    candidate_n_columns_bins,
    candidate_n_columns_n,
    'Number of Columns (Candidate Dataset)',
    output_filename + '-candidate-n-columns'
)

make_histogram(
    join_size_bins,
    join_size_n,
    'Join Size (Number of Columns)',
    output_filename + '-join-size'
)

make_histogram(
    query_size_bytes_bins,
    query_size_bytes_n,
    'Query Size (Bytes)',
    output_filename + '-query-size-bytes'
)

make_histogram(
    candidate_size_bytes_bins,
    candidate_size_bytes_n,
    'Candidate Size (Bytes)',
    output_filename + '-candidate-size-bytes'
)

# ML algorithms statistics

algorithms = ['xgboost', 'sgd', 'linear', 'random forest']

for i in range(len(algorithms)):
    algorithm = algorithms[i]

    stats_position = None
    try:
        stats_position = stats_file.index('Statistics for %s:'%algorithm)
    except ValueError:
        continue

    values = list()
    for j in range(stats_position + 2, stats_position + 10):
        line = stats_file[j]
        values.append(float(line[line.index('(')+1:line.index('%')]))

    make_before_after_histogram(
        values,
        output_filename + '-score-before-after-' + algorithm
    )
