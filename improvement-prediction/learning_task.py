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
import pickle
import json

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

    def _generate_models(self, data_splits_spec,
                         ml_algorithm_object,
                         ml_algorithm_name,
                         learning_target='gain_in_r2_score',
                         feature_ids=None,
                         save_model=True):
        """Executes a specific machine learning algorithm (e.g., random forests or linear regression) to predict a certain 
        learning_target. 

        Parameter data_splits_spec indicates how to split the data into training and test instances for cross validation (e.g., KFold).

        If feature_ids == None, all features are used to predict the targets; otherwise, only feature_ids are used.

        If save_model == True, save it with pickle.
        """
        
        if feature_ids:
            features = self.filter_learning_features(feature_ids)
        else:
            features = self.learning_features
        data_splits_spec.get_n_splits(features)
        targets = [item[learning_target] for item in self.learning_targets]
        
        i = 0
        models = []
        test_data_results = []
        for train_index, test_index in data_splits_spec.split(features):
            X_train, X_test = np.array(features)[train_index], np.array(features)[test_index]
            y_train, y_test = np.array(targets)[train_index], np.array(targets)[test_index]
            X_train, y_train = remove_outliers(X_train, y_train, zscore_threshold=0.5)
            X_test, y_test = remove_outliers(X_test, y_test, zscore_threshold=0.5)
            ml_algorithm_object.fit(X_train, y_train)            
                
            # generate predictions for test data
            predictions = ml_algorithm_object.predict(X_test)
            models.append(ml_algorithm_object)
            test_results = {'index_of_test_instances': test_index.tolist(),
                       'true_relative_gain_for_test_instances': y_test.tolist()}
            test_data_results.append(test_results)

            # save model (fitted ml_algorithm_object) and correspondint test data to disk
            if save_model:
                model_filename = 'finalized_model_' + ml_algorithm_name + '_for_test_fold_' + str(i) + '_predicting_' + learning_target + '.sav'
                pickle.dump(ml_algorithm_object, open(model_filename, 'wb'))
                test_data_filename = 'test_data_for_' + ml_algorithm_name + '_fold_' + str(i) + '_predicting_' + learning_target + '.json'
                json.dump(test_results, open(test_data_filename, 'w'))

            # the lines below help inspect the models, and how good they are
            ## performs feature selection in order to rank which features matter most for the model
            if ml_algorithm_name == 'random_forest':
                feature_importances = [(index, value) for index, value in enumerate(ml_algorithm_object.feature_importances_)]
                print([i[0] for i in sorted(feature_importances, key= lambda i: i[1], reverse=True)])
            elif ml_algorithm_name == 'linear_regression':
                mutual_information_univariate_selection(X_train, y_train)
            
            ## inspects the r2 score of the model over the test data
            print('how good this ' + ml_algorithm_name + ' is:', r2_score(y_test, predictions))
            
            ## computes error metrics between actual targets and predictions
            print('fold', i, 'SMAPE', compute_SMAPE(predictions, y_test), 'MSE', compute_MSE(predictions, y_test))

            ## contrasts actual targets (real values) and predictions (predicted values)
            #plot_scatterplot(y_test, predictions, 'predicted_r2_score_gains_fold_' + str(i) + '_' + ml_algorithm_name + '.png', 'Real values', 'Predicted values')

            #############
            i += 1
        return models, test_data_results

    def execute_linear_regression(self, n_splits, learning_target='gain_in_r2_score', feature_ids=None, save_model=True):
        """Performs linear regression with k-fold cross validation 
        (k = n_splits). 

        
        The learning target (parameter learning_target) needs to be specified, and it can be one of the following values:
        'decrease_in_mae', 'decrease_in_mse', 'decrease_in_med_ae', or 'gain_in_r2_score' (default).

        If feature_ids == None, all features are used to 
        predict the targets; otherwise, only feature_ids are used.

        If save_model == True, save it with pickle.
        """
        
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        lm = LinearRegression()
        return self._generate_models(kf, lm, 'linear_regression', learning_target, feature_ids, save_model)

    def execute_random_forest(self, n_splits, learning_target='gain_in_r2_score', feature_ids=None, save_model=True):
        """Performs random forest with k-fold cross validation 
        (k = n_splits). 

        The learning target (parameter learning_target) needs to be specified, and it can be one of the following values:
        'decrease_in_mae', 'decrease_in_mse', 'decrease_in_med_ae', or 'gain_in_r2_score' (default).

        If feature_ids == None, all features are used to 
        predict the targets; otherwise, only feature_ids are used.

        If save_model == True, save it with pickle.
        """
        
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        return self._generate_models(kf, rf, 'random_forest', learning_target, feature_ids, save_model)

