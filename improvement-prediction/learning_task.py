import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold 
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt

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
                 
    def execute_linear_regression(self, n_splits):
        kf = KFold(n_splits=n_splits, random_state=42)
        kf.get_n_splits(self.learning_data)
        i = 0
        for train_index, test_index in kf.split(self.learning_data):
            X_train, X_test = np.array(self.learning_data)[train_index], np.array(self.learning_data)[test_index]
            y_train, y_test = np.array(self.learning_targets)[train_index], np.array(self.learning_targets)[test_index]
            #rf = RandomForestRegressor(n_estimators=100, random_state=42)
            #rf.fit(X_train, y_train)
            #predictions = rf.predict(X_test)
            lm = LinearRegression()
            lm.fit(X_train, y_train)
            predictions = lm.predict(X_test)
            plt.scatter(y_test, predictions)
            plt.xlabel('Real values')
            plt.ylabel('Predicted values')
            plt.tight_layout()
            plt.savefig('predicted_r2_score_gains_fold_' + str(i) + '.png', dpi=300)
            plt.close()
            i += 1
            print('Mean squared error:', mean_squared_error(predictions, y_test))
            
