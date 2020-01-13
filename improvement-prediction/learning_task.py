import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
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
    def __init__(self, validation_type):
        """This class implements different models to learn relative performance gains 
        after executing data augmentation. 

        If validation_type == 'cross-validation', we generate different models with 
        k-fold cross-validation. Otherwise, if validation_type == 'training-test', 
        we use a file with training data and one with test data for the validation.
        """
        self.validation_type = validation_type
        self.learning_metadata = []
        self.learning_features = []
        self.learning_targets = []
                
    def add_learning_instance(self, learning_features, learning_target):
        """Stores features and learning target (relative gain) from an 
        augmentation instance
        """
        self.learning_features.append(learning_features)
        self.learning_targets.append(learning_target)

    def read_features_and_targets(self, augmentation_learning_filenames):
        """Reads metadata, features derived from training data, and corresponding targets 
        (relative performance gains after augmentation)
        """
        self.learning_metadata, self.learning_features, self.learning_targets = read_augmentation_learning_filename(augmentation_learning_filenames[0])
        if self.validation_type == 'training-test':
            test_metadata, test_features, test_targets = read_augmentation_learning_filename(augmentation_learning_filenames[1])
            self.training_size = len(self.learning_metadata)
            self.learning_metadata += test_metadata
            self.learning_features += test_features
            self.learning_targets += test_targets
              
    def filter_learning_features(self, feature_ids):
        """This method ensures that the learning will only use the features 
        indicated with feature_ids
        """
        return np.array(self.learning_features)[:,feature_ids]

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
        if type(data_splits_spec) is KFold:
            data_splits = [(train_index, test_index) for train_index, test_index in data_splits_spec.split(features)]
        else:
            data_splits = data_splits_spec

        targets = [item[learning_target] for item in self.learning_targets]
        
        i = 0
        models = []
        test_data_results = []
        for elem in data_splits:
            train_index = elem[0]
            test_index = elem[1]
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
            print('predictions', list(predictions))
            print('y_test', list(y_test))
            test_data_results.append(test_results)

            # save model (fitted ml_algorithm_object) and correspondint test data to disk
            if save_model:
                model_filename = 'finalized_model_' + ml_algorithm_name + '_for_test_fold_' + str(i) + '_predicting_' + learning_target + '.sav'
                pickle.dump(ml_algorithm_object, open(model_filename, 'wb'))
                test_data_filename = 'test_data_for_' + ml_algorithm_name + '_fold_' + str(i) + '_predicting_' + learning_target + '.json'
                json.dump(test_results, open(test_data_filename, 'w'))

            # the lines below help inspect the models, and how good they are
            ## performs feature selection in order to rank which features matter most for the model
            if ml_algorithm_name == 'random_forest' or ml_algorithm_name == 'decision_tree':
                feature_importances = sorted([(index, value) for index, value in enumerate(ml_algorithm_object.feature_importances_)], 
                                             key= lambda i: i[1], 
                                             reverse=True)
                print([(FEATURE_NAMES[i[0]], i[1]) for i in feature_importances])
            #elif ml_algorithm_name == 'linear_regression' or ml_algorithm_name == 'sgd_regressor':
            #    mutual_information_univariate_selection(X_train, y_train)
            
            ## inspects the r2 score of the model over the test data
            print('how good this ' + ml_algorithm_name + ' is:', r2_score(y_test, predictions))
            
            ## computes error metrics between actual targets and predictions
            print('fold', i, 'SMAPE', compute_SMAPE(predictions, y_test), 'MSE', compute_MSE(predictions, y_test))

            ## contrasts actual targets (real values) and predictions (predicted values)
            plot_scatterplot(y_test, predictions, 'predicted_r2_score_gains_fold_' + str(i) + '_' + ml_algorithm_name + '.png', 'Real values', 'Predicted values')

            ## contrasts actual targets (real values) and a few features that can be used as baselines
            plot_scatterplot(y_test, X_test[:,-1], 'containment_baseline_r2_score_gains_fold_' + str(i) + '_' + ml_algorithm_name + '.png', 'Real values', 'Containment')
            plot_scatterplot(y_test, X_test[:,-2], 'max_pearson_diff_baseline_r2_score_gains_fold_' + str(i) + '_' + ml_algorithm_name + '.png', 'Real values', 'Max pearson diff')
            #############
            i += 1
        return models, test_data_results

    def execute_linear_regression_cross_validation(self, n_splits, learning_target='gain_in_r2_score', feature_ids=None, save_model=True):
        """Performs linear regression with k-fold cross validation 
        (k = n_splits). 

        
        The learning target (parameter learning_target) needs to be specified, and it can be one of the following values:
        'decrease_in_mae', 'decrease_in_mse', 'decrease_in_med_ae', or 'gain_in_r2_score' (default).

        If feature_ids == None, all features are used to 
        predict the targets; otherwise, only feature_ids are used.

        If save_model == True, save it with pickle.
        """
        
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        lm = LinearRegression(normalize=True)
        return self._generate_models(kf, lm, 'linear_regression', learning_target, feature_ids, save_model)

    def execute_decision_trees_cross_validation(self, n_splits, learning_target='gain_in_r2_score', feature_ids=None, save_model=True):
        """Performs decision trees with k-fold cross validation 
        (k = n_splits). 

        The learning target (parameter learning_target) needs to be specified, and it can be one of the following values:
        'decrease_in_mae', 'decrease_in_mse', 'decrease_in_med_ae', or 'gain_in_r2_score' (default).

        If feature_ids == None, all features are used to 
        predict the targets; otherwise, only feature_ids are used.

        If save_model == True, save it with pickle.
        """        
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        dt = DecisionTreeRegressor(random_state=42)
        return self._generate_models(kf, dt, 'decision_tree', learning_target, feature_ids)

    def execute_random_forest(self, n_splits, learning_target='gain_in_r2_score', feature_ids=None, save_model=True):
        """Performs random forest with k-fold cross validation (k = n_splits) if self.validation_type == 'cross-validation'. 
        Otherwise, if self.validation_type == 'training-test', the random forest is trained over a fraction X of the data 
        and tested over a fraction 1 - X. 

        The learning target (parameter learning_target) needs to be specified, and it can be one of the following values:
        'decrease_in_mae', 'decrease_in_mse', 'decrease_in_med_ae', or 'gain_in_r2_score' (default).

        If feature_ids == None, all features are used to 
        predict the targets; otherwise, only feature_ids are used.
        The learning target (parameter learning_target) needs to be specified, and it can be one of the following values:
        'decrease_in_mae', 'decrease_in_mse', 'decrease_in_med_ae', or 'gain_in_r2_score' (default).

        If feature_ids == None, all features are used to 
        predict the targets; otherwise, only feature_ids are used.

        If save_model == True, save it with pickle.
        """
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        if self.validation_type == 'cross-validation':
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            return self._generate_models(kf, rf, 'random_forest', learning_target, feature_ids, save_model)
        elif self.validation_type == 'training-test':
            training_index = np.array([i for i in range(self.training_size)])
            test_index = np.array([i + self.training_size for i in range(len(self.learning_features) - self.training_size)])
            return self._generate_models([(training_index, test_index)], rf, 'random_forest', learning_target, feature_ids, save_model)
        return None, None

    def execute_sgd(self, n_splits, learning_target='gain_in_r2_score', feature_ids=None, save_model=True):
        """Performs a linear model that uses stochastic gradient descent. The execution uses k-fold cross validation (k = n_splits) if 
        self.validation_type == 'cross-validation'. Otherwise, if self.validation_type == 'training-test', the model is trained 
        over a fraction X of the data and tested over a fraction 1 - X. 

        The learning target (parameter learning_target) needs to be specified, and it can be one of the following values:
        'decrease_in_mae', 'decrease_in_mse', 'decrease_in_med_ae', or 'gain_in_r2_score' (default).

        If feature_ids == None, all features are used to 
        predict the targets; otherwise, only feature_ids are used.
        The learning target (parameter learning_target) needs to be specified, and it can be one of the following values:
        'decrease_in_mae', 'decrease_in_mse', 'decrease_in_med_ae', or 'gain_in_r2_score' (default).

        If feature_ids == None, all features are used to 
        predict the targets; otherwise, only feature_ids are used.

        If save_model == True, save it with pickle.
        """
        sgd = SGDRegressor(random_state=42)
        if self.validation_type == 'cross-validation':
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            return self._generate_models(kf, sgd, 'sgd_regressor', learning_target, feature_ids, save_model)
        elif self.validation_type == 'training-test':
            training_index = np.array([i for i in range(self.training_size)])
            test_index = np.array([i + self.training_size for i in range(len(self.learning_features) - self.training_size)])
            return self._generate_models([(training_index, test_index)], sgd, 'sgd_regressor', learning_target, feature_ids, save_model)
        return None, None
