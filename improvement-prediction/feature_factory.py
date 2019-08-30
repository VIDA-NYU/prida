import numpy as np
from scipy import stats

# TODO get more unidimensional features, using this metalearning
# document as reference: https://arxiv.org/pdf/1810.03548.pdf
class FeatureFactory:
    def get_dataset_features(self, data):
        self.data = data
        self.number_of_columns = self.data.shape[1]
        self.number_of_rows = self.data.shape[0]
        self.row_to_column_ratio = self.data.shape[0]/self.data.shape[1]
        self.number_of_numerical_columns = self.get_number_of_numerical_columns()
        self.means_of_numerical_columns = self.get_means_of_numerical_columns() 
        self.percentages_of_missing_values = self.get_percentages_of_missing_values()
        self.outlier_percentages_of_numerical_columns = self.get_outlier_percentages_of_numerical_columns()

        print('number of columns', self.number_of_columns, 'number of rows', self.number_of_rows,
              'ratio', self.row_to_column_ratio, 'number of numerical columns', self.number_of_numerical_columns,
              'means of numerical columns', self.means_of_numerical_columns, 'percentages of missing values',
              self.percentages_of_missing_values, 'outlier percentages of numerical columns',
              self.outlier_percentages_of_numerical_columns)

    def get_number_of_numerical_columns(self):
        return len([i for i in self.data.dtypes if i == np.int64 or i == np.float64])

    def get_means_of_numerical_columns(self):
        means = {}
        for column in self.data:
            if self.data[column].dtype == np.int64 or self.data[column].dtype == np.float64:
                means[column] = self.data[column].mean()
        return means

    def get_percentages_of_missing_values(self):
        percentage_of_missing_values = {}
        for column in self.data:
            percentage_of_missing_values[column] = self.data[column].isnull().sum()/self.data.shape[0]
        return percentage_of_missing_values

    def get_outlier_percentages_of_numerical_columns(self):
        outlier_percentages = {}
        for column in self.data:
            if self.data[column].dtype == np.int64 or self.data[column].dtype == np.float64:
                outlier_percentages[column] = len([i for i in stats.zscore(self.data[column]) if np.fabs(i) > 3])/self.data.shape[0]
        return outlier_percentages

    def get_skewness_of_numerical_columns(self):
        '''
        TODO
        '''
        pass
