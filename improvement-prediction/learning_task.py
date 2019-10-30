import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold 
from util.graphic_functions import *
from util.debug import *
from util.metrics import *
from constants import *
from util.feature_selection import *
from util.file_manager import *
from sklearn.preprocessing import MinMaxScaler

class LearningTask:
    def __init__(self):
        """This class implements different models to learn relative performance gains 
        after executing data augmentation
        """
        self.learning_metadata = []
        self.learning_features = []
        self.learning_targets = []

    def add_learning_instance(self, learning_features, learning_target):
        """Stores features and learning target (relative gain) from an 
        augmentation instance
        """
        self.learning_features.append(learning_features)
        self.learning_targets.append(learning_target)

    def read_features_and_targets(self, augmentation_learning_filename):
        """Reads metadata, features derived from training data, and corresponding targets 
        (relative performance gains after augmentation)
        """
        self.learning_metada, self.learning_features, self.learning_targets = read_augmentation_learning_filename(augmentation_learning_filename)

    def filter_learning_features(self, feature_ids):
        """This method ensures that the learning will only use the features 
        indicated with feature_ids
        """
        return self.learning_features[feature_ids]

    def execute_linear_regression(self, n_splits, learning_target='gain_in_r2_score', feature_ids=None):
        """Performs linear regression with k-fold cross validation 
        (k = n_splits). 

        
        The learning target (parameter learning_target) needs to be specified, and it can be one of the following values:
        'decrease_in_mae', 'decrease_in_mse', 'decrease_in_med_ae', or 'gain_in_r2_score' (default).

        If feature_ids == None, all features are used to 
        predict the targets; otherwise, only feature_ids are used.
        """
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        if feature_ids:
            features = self.filter_learning_features(feature_ids)
        else:
            features = self.learning_features
        kf.get_n_splits(features)
        targets = [item[learning_target] for item in self.learning_targets]
        
        i = 0
        models = []
        test_data_results = []
        for train_index, test_index in kf.split(features):
            X_train, X_test = np.array(features)[train_index], np.array(features)[test_index]
            y_train, y_test = np.array(targets)[train_index], np.array(targets)[test_index]

            X_train, y_train = remove_outliers(X_train, y_train, zscore_threshold=1)
            X_test, y_test = remove_outliers(X_test, y_test, zscore_threshold=1)
            lm = LinearRegression()
            lm.fit(X_train, y_train)
            models.append(rf)
            test_data_results.append({'index_of_test_instances': test_index,
                              'true_relative_gain_for_test_instances': y_test})
            predictions = lm.predict(X_test)

            # the lines below help inspect the models, and how good they are

            ## performs feature selection in order to rank which features matter most for the model
            mutual_information_univariate_selection(X_train, y_train)

            ## inspects the r2 score of the linear regression model over the test data
            print('how good is this linear regression model:', lm.score(X_test, y_test))

            ## computes error metrics between actual targets and predictions
            print('fold', i, 'SMAPE', compute_SMAPE(predictions, y_test), 'MSE', compute_MSE(predictions, y_test))

            ## contrasts actual targets (real values) and predictions (predicted values)
            plot_scatterplot(y_test, predictions, 'predicted_r2_score_gains_fold_' + str(i) + '_linear_regression.png', 'Real values', 'Predicted values')

            #############
            i += 1
            
        return models, test_data_results

    def execute_random_forest(self, n_splits, learning_target='gain_in_r2_score', feature_ids=None):
        """Performs random forest with k-fold cross validation 
        (k = n_splits). 

        The learning target (parameter learning_target) needs to be specified, and it can be one of the following values:
        'decrease_in_mae', 'decrease_in_mse', 'decrease_in_med_ae', or 'gain_in_r2_score' (default).

        If feature_ids == None, all features are used to 
        predict the targets; otherwise, only feature_ids are used.
        """        
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        if feature_ids:
            features = self.filter_learning_features(feature_ids)
        else:
            features = self.learning_features
        kf.get_n_splits(features)
        targets = [item[learning_target] for item in self.learning_targets]
        
        i = 0
        models = []
        test_data_results = []
        for train_index, test_index in kf.split(features):
            X_train, X_test = np.array(features)[train_index], np.array(features)[test_index]
            y_train, y_test = np.array(targets)[train_index], np.array(targets)[test_index]
            X_train, y_train = remove_outliers(X_train, y_train, zscore_threshold=0.5)
            X_test, y_test = remove_outliers(X_test, y_test, zscore_threshold=0.5)
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X_train, y_train)
            predictions = rf.predict(X_test)
            models.append(rf)
            test_data_results.append({'index_of_test_instances': test_index,
                              'true_relative_gain_for_test_instances': y_test})
            
            # the lines below help inspect the models, and how good they are

            ## performs feature selection in order to rank which features matter most for the model
            feature_importances = [(index, value) for index, value in enumerate(rf.feature_importances_)]
            print([i[0] for i in sorted(feature_importances, key= lambda i: i[1], reverse=True)])

            ## inspects the r2 score of the random forest model over the test data
            print('how good is this random forest model:', r2_score(y_test, predictions))
            
            ## computes error metrics between actual targets and predictions
            print('fold', i, 'SMAPE', compute_SMAPE(predictions, y_test), 'MSE', compute_MSE(predictions, y_test))

            ## contrasts actual targets (real values) and predictions (predicted values)
            plot_scatterplot(y_test, predictions, 'predicted_r2_score_gains_fold_' + str(i) + '_random_forest.png', 'Real values', 'Predicted values')

            #############
            i += 1
        return models, test_data_results

