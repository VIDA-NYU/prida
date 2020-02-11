"""
This script gets parameters
   1 - alpha \in R*: a threshold above which a certain gain in R2 score maps onto class 'good_gain', and onto class 'loss' otherwise
   2 - training filename: a filename for the dataset that we use to train the classifier
   3 - test filename: a filename for the use cases over which we want to test our model
and returns class predictions for each use case
"""

import sys
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

FEATURE_VECTOR = ['query_num_of_columns', 'query_num_of_rows', 'query_row_column_ratio','query_max_mean', 'query_max_outlier_percentage', 'query_max_skewness', 'query_max_kurtosis', 'query_max_unique', 'candidate_num_of_columns', 'candidate_num_rows', 'candidate_row_column_ratio', 'candidate_max_mean', 'candidate_max_outlier_percentage', 'candidate_max_skewness', 'candidate_max_kurtosis', 'candidate_max_unique', 'query_target_max_pearson', 'query_target_max_spearman', 'query_target_max_covariance', 'query_target_max_mutual_info', 'candidate_target_max_pearson', 'candidate_target_max_spearman', 'candidate_target_max_covariance', 'candidate_target_max_mutual_info', 'max_pearson_difference', 'containment_fraction']

GAIN_COLUMN_NAME = 'gain_in_r2_score'


def predict_class(alpha, training_filename, test_filename, feature_vector=FEATURE_VECTOR, gain_column=GAIN_COLUMN_NAME):
    """This function loads both datasets, creates classes for the instances in training based on
    the value of alpha, builds a classifier based on them, and predict classes for test instances
    """
    training_data = pd.read_csv(training_filename)
    training_data['class'] = ['good_gain' if row[gain_column] > alpha else 'loss' for index, row in training_data.iterrows()]
    X_train = training_data[feature_vector]
    y_train = training_data['class']
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    test_data = pd.read_csv(test_filename)
    X_test = test_data[feature_vector]
    return clf.predict(X_test)
    
if __name__ == '__main__':

    if len(sys.argv) != 4:
        print('Wrong number of parameters. Please pass, in this order, the threshold alpha, the filename for the model training data, and the filename for the use cases data')
        exit()

    alpha = float(sys.argv[1])
    if alpha < 0:
        print('Parameter alpha needs to be greater than or equal to zero')
        exit()

    training_filename = sys.argv[2]
    test_filename = sys.argv[3]
    print('predicted classes for use cases', predict_class(alpha, training_filename, test_filename))
