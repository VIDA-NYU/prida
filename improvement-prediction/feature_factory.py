import numpy as np
from scipy import stats

# reference document: https://arxiv.org/pdf/1810.03548.pdf
class FeatureFactory:
    def __init__(self, data):
        self.data = data

    def get_number_of_columns(self):
        return self.data.shape[1]

    def get_number_of_rows(self):
        return self.data.shape[0]

    def get_row_to_column_ratio(self):
        return self.data.shape[0]/self.data.shape[1]
        
    def get_number_of_numerical_columns(self):
        return len([i for i in self.data.dtypes if i == np.int64 or i == np.float64])

    def get_means_of_numerical_columns(self):
        mean = {}
        for column in self.data:
            if self.data[column].dtype == np.int64 or self.data[column].dtype == np.float64:
                mean[column] = self.data[column].mean()
        return mean

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
        skewness = {}
        for column in self.data:
            if self.data[column].dtype == np.int64 or self.data[column].dtype == np.float64:
                skewness[column] = self.data[column].skew(skipna=True)
        return skewness

    
    def get_kurtosis_of_numerical_columns(self):
        kurtosis = {}
        for column in self.data:
            if self.data[column].dtype == np.int64 or self.data[column].dtype == np.float64:
                kurtosis[column] = self.data[column].kurtosis(skipna=True)
        return kurtosis

    def get_number_of_unique_values_of_numerical_columns(self):
        number_of_unique_values = {}
        for column in self.data:
            if self.data[column].dtype == np.int64 or self.data[column].dtype == np.float64:
                number_of_unique_values[column] = self.data[column].nunique()
                print(column, number_of_unique_values[column])
        return number_of_unique_values

    
