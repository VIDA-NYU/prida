TRAINING_FILENAME = '../../classification/training-simplified-data-generation.csv'
THETA = 0.0
SEPARATOR = ','
RENAME_NUMERICAL = False
MEAN_DATA_IMPUTATION = True

CLASS_ATTRIBUTE_NAME = 'class_pos_neg'

DATASET_FEATURES = ['query_num_of_columns', 'query_num_of_rows', 'query_row_column_ratio', 'query_max_mean', 'query_max_outlier_percentage', 
                    'query_max_skewness', 'query_max_kurtosis', 'query_max_unique', 'candidate_num_of_columns', 'candidate_num_rows', 
                    'candidate_row_column_ratio', 'candidate_max_mean', 'candidate_max_outlier_percentage', 'candidate_max_skewness',
                    'candidate_max_kurtosis', 'candidate_max_unique']
QUERY_TARGET_FEATURES = ['query_target_max_pearson', 'query_target_max_spearman', 'query_target_max_covariance', 'query_target_max_mutual_info']
CANDIDATE_TARGET_FEATURES = ['candidate_target_max_pearson', 'candidate_target_max_spearman',
                             'candidate_target_max_covariance', 'candidate_target_max_mutual_info']
DATASET_DATASET_FEATURES = ['containment_fraction']

FEATURES = DATASET_FEATURES + QUERY_TARGET_FEATURES + CANDIDATE_TARGET_FEATURES + DATASET_DATASET_FEATURES

GAIN_ATTRIBUTE_NAME = 'gain_in_r2_score'

