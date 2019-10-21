import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold 
from util.graphic_functions import *
from util.debug import *
from util.metrics import *
from util.feature_selection import *
from sklearn.preprocessing import MinMaxScaler

SEPARATOR = ','

class LearningTask:
    def __init__(self):
        self.learning_data = []
        self.learning_targets = []

    def add_learning_instance(self, learning_features, learning_target):
        self.learning_data.append(learning_features)
        self.learning_targets.append(learning_target)

    def dump_learning_instances(self, data_filename):
        with open(data_filename, 'w') as f:
            for features, target in zip(self.learning_data, self.learning_targets):
                output_string = ','.join([str(i) for i in features]) + ',' + str(target) + '\n'
                f.write(output_string)

    def read_data(self, augmentation_learning_filename):
        with open(augmentation_learning_filename, 'r') as f:
             for line in f:
                 fields = [float(i) for i in line.strip().split(SEPARATOR)]
                 #assuming that the relative r-squared gain is in fields[-1]
                 self.learning_data.append(fields[:-1])
                 self.learning_targets.append(fields[-1])
             self.learning_data = pd.DataFrame(self.learning_data)
             self.learning_targets = pd.DataFrame(self.learning_targets)

    def filter_learning_data(self, feature_ids):
        return self.learning_data[feature_ids]

    def execute_linear_regression(self, n_splits, feature_ids=None):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        if feature_ids:
            data = self.filter_learning_data(feature_ids)
        else:
            data = self.learning_data
        kf.get_n_splits(data)
        i = 0
        for train_index, test_index in kf.split(data):
            X_train, X_test = np.array(data)[train_index], np.array(data)[test_index]
            y_train, y_test = np.array(self.learning_targets)[train_index], np.array(self.learning_targets)[test_index]

            X_train, y_train = remove_outliers(X_train, y_train, zscore_threshold=1)
            X_test, y_test = remove_outliers(X_test, y_test, zscore_threshold=1)
            lm = LinearRegression()
            lm.fit(X_train, y_train)            
            predictions = lm.predict(X_test)
            mutual_information_univariate_selection(X_train, y_train)
            print('how good is this linear regression model:', lm.score(X_test, y_test))
            print('fold', i, 'SMAPE', compute_SMAPE(predictions, y_test), 'MSE', compute_MSE(predictions, y_test))
            plot_scatterplot(y_test, predictions, 'predicted_r2_score_gains_fold_' + str(i) + '_linear_regression.png', 'Real values', 'Predicted values')
            i += 1

    def execute_random_forest(self, n_splits, feature_ids=None):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        if feature_ids:
            data = self.filter_learning_data(feature_ids)
        else:
            data = self.learning_data
        kf.get_n_splits(data)
        i = 0
        for train_index, test_index in kf.split(data):
            X_train, X_test = np.array(data)[train_index], np.array(data)[test_index]
            y_train, y_test = np.array(self.learning_targets)[train_index], np.array(self.learning_targets)[test_index]

            #print('averages medians and stds before', np.mean(y_train), np.median(y_train), np.std(y_train), np.mean(y_test), np.median(y_test), np.std(y_test))
            X_train, y_train = remove_outliers(X_train, y_train, zscore_threshold=0.5)
            X_test, y_test = remove_outliers(X_test, y_test, zscore_threshold=0.5)
            #print('averages medians and stds after', np.mean(y_train), np.median(y_train), np.std(y_train), np.mean(y_test), np.median(y_train), np.std(y_test))
            
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X_train, y_train)
            feature_importances = [(index, value) for index, value in enumerate(rf.feature_importances_)]
            print([i[0] for i in sorted(feature_importances, key= lambda i: i[1], reverse=True)])
            predictions = rf.predict(X_test)
            print('how good is this random forest model:', r2_score(y_test, predictions))
            print('fold', i, 'MSE', compute_MSE(predictions, y_test))
            plot_scatterplot(y_test, predictions, 'predicted_r2_score_gains_fold_' + str(i) + '_random_forest.png', 'Real values', 'Predicted values')
            i += 1

