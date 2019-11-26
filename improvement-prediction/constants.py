SEPARATOR = ','
GAUSSIAN_OUTLIER_THRESHOLD = 3
NUMBER_OF_FIELDS_IN_TEST_AUGMENTATION_INSTANCE = 4
NUMBER_OF_SPARK_REPARTITIONS = 200
QUERY_FILENAME_ID = 0
TARGET_NAME_ID = 1
CANDIDATE_FILENAME_ID = 2
GAIN_IN_R2_SCORE_ID = -1
DECREASE_IN_MEDIAN_ABSOLUTE_ERROR_ID = -2
DECREASE_IN_MEAN_SQUARED_ERROR_ID = -3
DECREASE_IN_MEAN_ABSOLUTE_ERROR_ID = -4


color_dict = {'query_max_kurtosis': 'lightblue', 'query_max_mean': 'lightblue', 'query_row_column_ratio': 'lightblue', 'query_num_of_columns': 'lightblue', 'query_max_outlier_percentage': 'lightblue', 'query_max_unique': 'lightblue', 'query_num_rows': 'lightblue', 'query_target_max_mutual_info': 'blue', 'query_target_max_covariance': 'blue', 'query_target_max_pearson': 'blue', 'query_target_max_spearman': 'blue', 'candidate_max_kurtosis': 'pink', 'candidate_max_unique': 'pink', 'candidate_num_rows': 'pink', 'candidate_max_mean': 'pink', 'candidate_row_column_ratio': 'pink', 'candidate_max_skewness': 'pink', 'candidate_max_outlier_percentage': 'pink', 'candidate_num_of_columns': 'pink', 'candidate_max_skewness': 'pink', 'candidate_target_max_mutual_info': 'red', 'candidate_target_max_spearman': 'red', 'candidate_target_max_pearson': 'red', 'candidate_target_max_covariance': 'red', 'containment_fraction': 'green', 'max_pearson_difference': 'yellow'}


FEATURE_NAMES = ['query_num_of_columns','query_num_of_rows','query_row_column_ratio','query_max_mean','query_max_outlier_percentage','query_max_skewness','query_max_kurtosis','query_max_unique','candidate_num_of_columns','candidate_num_rows','candidate_row_column_ratio','candidate_max_mean','candidate_max_outlier_percentage','candidate_max_skewness','candidate_max_kurtosis','candidate_max_unique','query_target_max_pearson','query_target_max_spearman','query_target_max_covariance','query_target_max_mutual_info','candidate_target_max_pearson','candidate_target_max_spearman','candidate_target_max_covariance','candidate_target_max_mutual_info','max_pearson_difference', 'containment_fraction']
CONTAINMENT_FEATURE_ID = -1
DIFFERENCE_IN_PEARSON_FEATURE_ID = -2
