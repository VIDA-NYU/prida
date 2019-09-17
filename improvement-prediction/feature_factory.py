import numpy as np
from scipy import stats
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics.cluster import normalized_mutual_info_score

# reference document: https://arxiv.org/pdf/1810.03548.pdf
class FeatureFactory:
    def __init__(self, data):
        self.data = data
        
    def _is_integer(self, column_name):
        return self.data[column_name].dtype == np.int64 or self.data[column_name].dtype == np.int32

    def _is_float(self, column_name):
        return self.data[column_name].dtype == np.float64 or self.data[column_name].dtype == np.float32
    
    def _is_numerical(self, column_name):
        return self.data[column_name].dtype == np.int64 or self.data[column_name].dtype == np.int32 or self.data[column_name].dtype == np.float64 or self.data[column_name].dtype == np.float32
    
    def get_number_of_columns(self):
        return self.data.shape[1]

    def get_number_of_rows(self):
        return self.data.shape[0]

    def get_row_to_column_ratio(self):
        return self.data.shape[0]/self.data.shape[1]
        
    def get_number_of_numerical_columns(self):
        return len([column for column in self.data if self._is_numerical(column)])

    def get_means_of_numerical_columns(self):
        mean = {}
        for column in self.data:
            if self._is_numerical(column):
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
            if self._is_numerical(column): 
                outlier_percentages[column] = len([i for i in stats.zscore(self.data[column]) if np.fabs(i) > 3])/self.data.shape[0]
        return outlier_percentages

    def get_skewness_of_numerical_columns(self):
        skewness = {}
        for column in self.data:
            if self._is_numerical(column): 
                skewness[column] = self.data[column].skew(skipna=True)
        return skewness

    
    def get_kurtosis_of_numerical_columns(self):
        kurtosis = {}
        for column in self.data:
            if self._is_numerical(column): 
                kurtosis[column] = self.data[column].kurtosis(skipna=True)
        return kurtosis

    def get_number_of_unique_values_of_numerical_columns(self):
        number_of_unique_values = {}
        for column in self.data:
            if self._is_numerical(column): 
                number_of_unique_values[column] = self.data[column].nunique()
        return number_of_unique_values

    def get_individual_metrics(self, func=max):
        metrics = [self.get_number_of_columns(),
                   self.get_number_of_rows(),
                   self.get_row_to_column_ratio(),
                   self.get_number_of_numerical_columns()]
        metrics.append(func(self.get_means_of_numerical_columns().values()))
        metrics.append(func(self.get_percentages_of_missing_values().values()))
        metrics.append(func(self.get_outlier_percentages_of_numerical_columns().values()))
        metrics.append(func(self.get_skewness_of_numerical_columns().values()))
        metrics.append(func(self.get_kurtosis_of_numerical_columns().values()))
        metrics.append(func(self.get_number_of_unique_values_of_numerical_columns().values()))
        return metrics
        
    def get_pearson_correlations(self):
        corrs = self.data.corr(method='pearson')
        correlations = []
        for index1, column1 in enumerate(corrs):
            for index2, column2 in enumerate(corrs):
                if column1 != column2 and index1 < index2:
                    correlations.append(((column1, column2), corrs[column1][column2]))
        return correlations

    def get_spearman_correlations(self):
        corrs = self.data.corr(method='spearman')
        correlations = []
        for index1, column1 in enumerate(corrs):
            for index2, column2 in enumerate(corrs):
                if column1 != column2 and index1 < index2:
                    correlations.append(((column1, column2), corrs[column1][column2]))
        return correlations

    def get_kendall_tau_correlations(self):
        corrs = self.data.corr(method='kendall')
        correlations = []
        for index1, column1 in enumerate(corrs):
            for index2, column2 in enumerate(corrs):
                if column1 != column2 and index1 < index2:
                    correlations.append(((column1, column2), corrs[column1][column2]))
        return correlations    

    def get_covariances(self):
        covs = self.data.cov()
        covariances = []
        for index1, column1 in enumerate(covs):
            for index2, column2 in enumerate(covs):
                if column1 != column2 and index1 < index2:
                    covariances.append(((column1, column2), covs[column1][column2]))
        return covariances

    def get_entropy_levels_int(self):
        entropy_levels_int = {}
        for column in self.data:
            if self._is_integer(column):
                value, counts = np.unique(self.data[column], return_counts=True)            
                entropy_levels_int[column] = stats.entropy(counts)
        return entropy_levels_int

    #TODO does it make sense to treat ints and floats differently here?
    def get_entropy_levels_float(self):
        entropy_levels_float = {}
        for column in self.data:
            if self._is_integer(column):
                original_values = self.data[column]
                min_max_scaler = preprocessing.MinMaxScaler()
                scaled_values = min_max_scaler.fit_transform(original_values)
                entropy_levels_float[column] = stats.entropy(np.histogram(scaled_values)[0])
        return entropy_levels_float

    def get_normalized_mutual_information(self):
        mutual_infos = []
        for index1, column1 in enumerate(self.data):
            for index2, column2 in enumerate(self.data):
                if column1 != column2 and index1 < index2 and self._is_numerical(column1) and self._is_numerical(column2):
                    norm_mutual_info = normalized_mutual_info_score(self.data[column1], self.data[column2])
                    mutual_infos.append(((column1, column2), norm_mutual_info))
        return mutual_infos

    #TODO: implement concentration, ANOVA p-value
