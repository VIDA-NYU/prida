import numpy as np
from scipy import stats, cov
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics.cluster import normalized_mutual_info_score

GAUSSIAN_OUTLIER_THRESHOLD = 3

def max_in_modulus(values):
    return max(values, key=abs)
    
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
                outlier_percentages[column] = len([i for i in stats.zscore(self.data[column]) if np.fabs(i) > GAUSSIAN_OUTLIER_THRESHOLD])/self.data.shape[0]
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

    def get_entropy_levels_float(self):
        entropy_levels = {}
        for column in self.data:
            if self._is_float(column):
                original_values = self.data[column]
                original_values = original_values.values.reshape(1, -1)
                min_max_scaler = preprocessing.MinMaxScaler()
                scaled_values = min_max_scaler.fit_transform(original_values)
                entropy_levels[column] = stats.entropy(np.histogram(scaled_values)[0])
        return entropy_levels
    
    def get_entropy_levels_integer(self):
        entropy_levels = {}
        for column in self.data:
            if self._is_integer(column):
                value, counts = np.unique(self.data[column], return_counts=True)            
                entropy_levels[column] = stats.entropy(counts)
        return entropy_levels

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
        entropy_levels_float = self.get_entropy_levels_float()
        metrics.append(func(entropy_levels_float.values())) if entropy_levels_float else metrics.append(0.0)
        entropy_levels_int = self.get_entropy_levels_integer()
        metrics.append(func(entropy_levels_int.values())) if entropy_levels_int else metrics.append(0)
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

    def get_normalized_mutual_information(self):
        mutual_infos = []
        for index1, column1 in enumerate(self.data):
            for index2, column2 in enumerate(self.data):
                if column1 != column2 and index1 < index2 and self._is_numerical(column1) and self._is_numerical(column2):
                    norm_mutual_info = normalized_mutual_info_score(self.data[column1], self.data[column2])
                    mutual_infos.append(((column1, column2), norm_mutual_info))
        return mutual_infos

    def get_pairwise_metrics(self, func=max):
        metrics = []
        pearson = func([i[1] for i in self.get_pearson_correlations()])
        spearman = func([i[1] for i in self.get_spearman_correlations()])
        kendalltau = func([i[1] for i in self.get_kendall_tau_correlations()])
        covariance = func([i[1] for i in self.get_covariances()])
        mutual_info = func([i[1] for i in self.get_normalized_mutual_information()])
        metrics.append(pearson) if not np.isnan(pearson) else metrics.append(0.0)
        metrics.append(spearman) if not np.isnan(spearman) else metrics.append(0.0)
        metrics.append(kendalltau) if not np.isnan(kendalltau) else metrics.append(0.0)
        metrics.append(covariance) if not np.isnan(covariance) else metrics.append(0.0)
        metrics.append(mutual_info) if not np.isnan(mutual_info) else metrics.append(0.0)
        return metrics

    def get_pearson_correlations_with_target(self, target_column_name):
        correlations = {}
        for column in self.data:
            if column != target_column_name and self._is_numerical(column) and self._is_numerical(target_column_name):
                coefficient, pvalue = stats.pearsonr(self.data[column], self.data[target_column_name])
                #for now, i am ignoring the pvalues
                if not np.isnan(coefficient):
                    correlations[column] = coefficient
        return correlations

    def get_max_pearson_wrt_target(self, target_column_name):
        correlations = self.get_pearson_correlations_with_target(target_column_name)
        return max_in_modulus(correlations.values())
        
    def compute_difference_in_pearsons_wrt_target(self, max_in_modulus_pearson, target_column_name):
        return self.get_max_pearson_wrt_target(target_column_name) - max_in_modulus_pearson

    def get_spearman_correlations_with_target(self, target_column_name):
        correlations = {}
        for column in self.data:
            if column != target_column_name and self._is_numerical(column) and self._is_numerical(target_column_name):
                coefficient, pvalue = stats.spearmanr(self.data[column], self.data[target_column_name])
                #for now, i am ignoring the pvalues
                if not np.isnan(coefficient):
                    correlations[column] = coefficient
        return correlations

    def get_kendall_tau_correlations_with_target(self, target_column_name):
        correlations = {}
        for column in self.data:
            if column != target_column_name and self._is_numerical(column) and self._is_numerical(target_column_name):
                coefficient, pvalue = stats.kendalltau(self.data[column], self.data[target_column_name])
                #for now, i am ignoring the pvalues
                if not np.isnan(coefficient):
                    correlations[column] = coefficient
        return correlations

    def get_covariances_with_target(self, target_column_name):
        covariances = {}
        for column in self.data:
            if column != target_column_name and self._is_numerical(column) and self._is_numerical(target_column_name):
                covariance = cov(self.data[column], self.data[target_column_name])[0,1]
                if not np.isnan(covariance):
                    covariances[column] = covariance
        return covariances

    def get_normalized_mutual_information_with_target(self, target_column_name):
        mutual_infos = {}
        for column in self.data:
            if column != target_column_name and self._is_numerical(column) and self._is_numerical(target_column_name):
                norm_mutual_info = normalized_mutual_info_score(self.data[column], self.data[target_column_name])
                mutual_infos[column] = norm_mutual_info
        return mutual_infos

    def get_pairwise_metrics_with_target(self, target_column_name, func=max):
        metrics = []
        pearson = func(self.get_pearson_correlations_with_target(target_column_name).values())
        spearman = func(self.get_spearman_correlations_with_target(target_column_name).values())
        kendalltau = func(self.get_kendall_tau_correlations_with_target(target_column_name).values())
        covariance = func(self.get_covariances_with_target(target_column_name).values())
        mutual_info = func(self.get_normalized_mutual_information_with_target(target_column_name).values())
        #TODO refactor to avoid repetition
        metrics.append(pearson) if not np.isnan(pearson) else metrics.append(0.0)
        metrics.append(spearman) if not np.isnan(spearman) else metrics.append(0.0)
        metrics.append(kendalltau) if not np.isnan(kendalltau) else metrics.append(0.0)
        metrics.append(covariance) if not np.isnan(covariance) else metrics.append(0.0)
        metrics.append(mutual_info) if not np.isnan(mutual_info) else metrics.append(0.0)
        return metrics
