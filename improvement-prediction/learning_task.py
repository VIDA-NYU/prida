import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold 
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import f_regression, mutual_info_regression
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
        #features_and_results = []
        for train_index, test_index in kf.split(self.learning_data):
            X_train, X_test = np.array(self.learning_data)[train_index], np.array(self.learning_data)[test_index]
            y_train, y_test = np.array(self.learning_targets)[train_index], np.array(self.learning_targets)[test_index]
            print('number of features', len(X_train[0]))
            lm = LinearRegression()
            model = lm.fit(X_train, y_train)
            predictions = lm.predict(X_test)
            f_test, _ = f_regression(X_test, y_test)
            mi = mutual_info_regression(X_test, y_test)
            mi /= np.max(mi)
            feature_gains = []
            #TODO (1) plot best regressors against target (2) understand cases to which the errors are very high (3) analyze the signal alone
            #for index, elem in enumerate(mi):
            #    feature_gains.append((index, mi[index]))
            #print(sorted(feature_gains, key=lambda x: x[1], reverse=True))
            
            #for index, elem in enumerate(predictions):
            #    features_and_results.append([X_test[index], y_test[index], predictions[index], (y_test[index] - predictions[index])**2])
            #print(sorted(features_and_results, key=lambda x: x[3]))

            


# f_test, _ = f_regression(X, y)
# f_test /= np.max(f_test)

# mi = mutual_info_regression(X, y)
# mi /= np.max(mi)

# plt.figure(figsize=(15, 5))
# for i in range(3):
#     plt.subplot(1, 3, i + 1)
#     plt.scatter(X[:, i], y, edgecolor='black', s=20)
#     plt.xlabel("$x_{}$".format(i + 1), fontsize=14)
#     if i == 0:
#         plt.ylabel("$y$", fontsize=14)
#     plt.title("F-test={:.2f}, MI={:.2f}".format(f_test[i], mi[i]),
#               fontsize=16)
# plt.show()
            
            # plt.scatter(y_test, predictions)
            # plt.xlabel('Real values')
            # plt.ylabel('Predicted values')
            # plt.tight_layout()
            # plt.savefig('predicted_r2_score_gains_fold_' + str(i) + '.png', dpi=300)
            # plt.close()
            # i += 1
#            print(mean_squared_error(predictions, y_test))
            
