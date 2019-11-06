import numpy as np
from scipy import stats, cov
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics.cluster import normalized_mutual_info_score
from constants import *

def max_in_modulus(values):
    return max(values, key=abs)
    
# reference document: https://arxiv.org/pdf/1810.03548.pdf
class FeatureFactory:
    def __init__(self, data):
        """Given a pandas dataset, this class computes a variety of features over 
        its columns and their combinations
        """
        self.data = data
        
    def _is_integer(self, column_name):
        """Given the name of a column in the dataset, this method checks if its type is integer
        """
        return self.data[column_name].dtype == np.int64 or self.data[column_name].dtype == np.int32

    def _is_float(self, column_name):
        """Given the name of a column in the dataset, this method checks if its type is real
        """
        return self.data[column_name].dtype == np.float64 or self.data[column_name].dtype == np.float32
    
    def _is_numerical(self, column_name):
        """Given the name of a column in the dataset, this method checks if it is numerical (either real or integer)
        """
        return self.data[column_name].dtype == np.int64 or self.data[column_name].dtype == np.int32 or self.data[column_name].dtype == np.float64 or self.data[column_name].dtype == np.float32
        
    def get_number_of_columns(self):
        """Returns the number of columns in the dataset
        """
        return self.data.shape[1]

    def get_number_of_rows(self):
        """Returns the number of rows in the dataset
        """
        return self.data.shape[0]

    def get_row_to_column_ratio(self):
        """Returns the row-to-column ratio of the dataset
        """
        return self.data.shape[0]/self.data.shape[1]
        
    def get_number_of_numerical_columns(self):
        """Returns the number of numerical columns in the dataset
        """
        return len([column for column in self.data if self._is_numerical(column)])

    def get_means_of_numerical_columns(self):
        """Computes the mean of every numerical column in the dataset
        """
        mean = {}
        for column in self.data:
            if self._is_numerical(column):
                mean[column] = self.data[column].mean()
        return mean

    def get_percentages_of_missing_values(self):
        """Computes the percentage of missing values of every column in 
        the dataset
        """
        percentage_of_missing_values = {}
        for column in self.data:
            percentage_of_missing_values[column] = self.data[column].isnull().sum()/self.data.shape[0]
        return percentage_of_missing_values

    def get_outlier_percentages_of_numerical_columns(self):
        """Computes the percentage of outlier values of every numerical
        column in the dataset. Outliers are values whose corresponding zscores are above 
        a certain threshold (in modulus). TODO Check if this is the best way of computing outliers in this 
        dataset, as the columns are not necessarily normally distributed
        """
        outlier_percentages = {}
        for column in self.data:
            if self._is_numerical(column): 
                outlier_percentages[column] = len([i for i in stats.zscore(self.data[column]) if np.fabs(i) > GAUSSIAN_OUTLIER_THRESHOLD])/self.data.shape[0]
        return outlier_percentages

    def get_skewness_of_numerical_columns(self):
        """Computes the skewness of every numerical column in the dataset
        """
        skewness = {}
        for column in self.data:
            if self._is_numerical(column): 
                skewness[column] = self.data[column].skew(skipna=True)
        return skewness

    
    def get_kurtosis_of_numerical_columns(self):
        """Computes the kurtosis of every numerical column in the dataset
        """        
        kurtosis = {}
        for column in self.data:
            if self._is_numerical(column): 
                kurtosis[column] = self.data[column].kurtosis(skipna=True)
        return kurtosis

    def get_number_of_unique_values_of_numerical_columns(self):
        """Computes number of unique values of every column in the dataset
        TODO this feature would probably be more representative if we computed 
        the percentage of unique values
        """
        number_of_unique_values = {}
        for column in self.data:
            if self._is_numerical(column): 
                number_of_unique_values[column] = self.data[column].nunique()
        return number_of_unique_values

    def get_entropy_levels_float(self):
        """Computes the entropy level of every real (float) column of the dataset
        """
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
        """Computes the entropy level of every integer column of the dataset
        """        
        entropy_levels = {}
        for column in self.data:
            if self._is_integer(column):
                value, counts = np.unique(self.data[column], return_counts=True)            
                entropy_levels[column] = stats.entropy(counts)
        return entropy_levels

    def get_individual_features(self, func=max):
        """Computes all features of the dataset that concern columns individually, 
        i.e., no features that have to do with relationships across different columns. 
        The method returns for example the maximum (in modulus) of them --- it depends 
        on parameter func
        """
        features = [self.get_number_of_columns(),
                   self.get_number_of_rows(),
                   self.get_row_to_column_ratio(),
                   self.get_number_of_numerical_columns()]
        features.append(func(self.get_means_of_numerical_columns().values()))
        features.append(func(self.get_percentages_of_missing_values().values()))
        features.append(func(self.get_outlier_percentages_of_numerical_columns().values()))
        features.append(func(self.get_skewness_of_numerical_columns().values()))
        features.append(func(self.get_kurtosis_of_numerical_columns().values()))
        features.append(func(self.get_number_of_unique_values_of_numerical_columns().values()))
        entropy_levels_float = self.get_entropy_levels_float()
        features.append(func(entropy_levels_float.values())) if entropy_levels_float else features.append(0.0)
        entropy_levels_int = self.get_entropy_levels_integer()
        features.append(func(entropy_levels_int.values())) if entropy_levels_int else features.append(0)
        return features
        
    def get_pearson_correlations(self):
        """Computes pearson correlation for every pair of numerical
        columns of the dataset
        """
        corrs = self.data.corr(method='pearson')
        correlations = []
        for index1, column1 in enumerate(corrs):
            for index2, column2 in enumerate(corrs):
                if column1 != column2 and index1 < index2:
                    correlations.append(((column1, column2), corrs[column1][column2]))
        return correlations
    
    def get_spearman_correlations(self):
        """Computes spearman correlation for every pair of numerical
        columns of the dataset
        """        
        corrs = self.data.corr(method='spearman')
        correlations = []
        for index1, column1 in enumerate(corrs):
            for index2, column2 in enumerate(corrs):
                if column1 != column2 and index1 < index2:
                    correlations.append(((column1, column2), corrs[column1][column2]))
        return correlations

    def get_kendall_tau_correlations(self):
        """Computes kendall tau correlation for every pair of numerical
        columns of the dataset
        """        
        corrs = self.data.corr(method='kendall')
        correlations = []
        for index1, column1 in enumerate(corrs):
            for index2, column2 in enumerate(corrs):
                if column1 != column2 and index1 < index2:
                    correlations.append(((column1, column2), corrs[column1][column2]))
        return correlations    

    def get_covariances(self):
        """Computes the covariance between every pair of numerical
        columns of the dataset
        """        
        covs = self.data.cov()
        covariances = []
        for index1, column1 in enumerate(covs):
            for index2, column2 in enumerate(covs):
                if column1 != column2 and index1 < index2:
                    covariances.append(((column1, column2), covs[column1][column2]))
        return covariances

    def get_normalized_mutual_information(self):
        """Computes the mutual information between every pair of numerical
        columns of the dataset
        """        
        mutual_infos = []
        for index1, column1 in enumerate(self.data):
            for index2, column2 in enumerate(self.data):
                if column1 != column2 and index1 < index2 and self._is_numerical(column1) and self._is_numerical(column2):
                    norm_mutual_info = normalized_mutual_info_score(self.data[column1], self.data[column2])
                    mutual_infos.append(((column1, column2), norm_mutual_info))
        return mutual_infos

    def get_pairwise_features(self, func=max):
        """Computes all features of the dataset that concern pairs of columns, returning for 
        example the maximum (in modulus) of them --- it depends on parameter func
        """        
        features = []
        pearson = func([i[1] for i in self.get_pearson_correlations()])
        spearman = func([i[1] for i in self.get_spearman_correlations()])
        kendalltau = func([i[1] for i in self.get_kendall_tau_correlations()])
        covariance = func([i[1] for i in self.get_covariances()])
        mutual_info = func([i[1] for i in self.get_normalized_mutual_information()])
        features.append(pearson) if not np.isnan(pearson) else features.append(0.0)
        features.append(spearman) if not np.isnan(spearman) else features.append(0.0)
        features.append(kendalltau) if not np.isnan(kendalltau) else features.append(0.0)
        features.append(covariance) if not np.isnan(covariance) else features.append(0.0)
        features.append(mutual_info) if not np.isnan(mutual_info) else features.append(0.0)
        return features

    def get_pearson_correlations_with_target(self, target_name):
        """For every numerical column of the dataset, this method computes its pearson correlation 
        with respect to the target column
        """
        correlations = {}
        for column in self.data:
            if column != target_name and self._is_numerical(column) and self._is_numerical(target_name):
                coefficient, pvalue = stats.pearsonr(self.data[column], self.data[target_name])
                #for now, i am ignoring the pvalues
                if not np.isnan(coefficient):
                    #if 'nan', either the column or the target_name is constant. using zero in this case
                    correlations[column] = coefficient
                else:
                    correlations[column] = 0.0
        return correlations

    def get_max_pearson_wrt_target(self, target_name):
        """Given the pearson correlations between every numerical column of the dataset and the target column, 
        this method returns the maximum (in modulus) of them
        """
        correlations = self.get_pearson_correlations_with_target(target_name)
        return max_in_modulus(correlations.values())
        
    def compute_difference_in_pearsons_wrt_target(self, max_in_modulus_pearson, target_name):
        """Given the maximum pearson correlation (in modulus) between the numerical columns of this dataset 
        and the target column, and the maximum pearson correlation (in modulus) between the numerical columns of 
        another dataset and the same target column, this method returns their difference 
        """
        return self.get_max_pearson_wrt_target(target_name) - max_in_modulus_pearson

    def get_spearman_correlations_with_target(self, target_name):
        """For every numerical column of the dataset, this method computes its spearman correlation 
        with respect to the target column
        """        
        correlations = {}
        for column in self.data:
            if column != target_name and self._is_numerical(column) and self._is_numerical(target_name):
                coefficient, pvalue = stats.spearmanr(self.data[column], self.data[target_name])
                #for now, i am ignoring the pvalues
                if not np.isnan(coefficient):
                    correlations[column] = coefficient
                else:
                    correlations[column] = 0.0
        return correlations

    def get_kendall_tau_correlations_with_target(self, target_name):
        """For every numerical column of the dataset, this method computes its kendall tau correlation 
        with respect to the target column
        """
        correlations = {}
        for column in self.data:
            if column != target_name and self._is_numerical(column) and self._is_numerical(target_name):
                coefficient, pvalue = stats.kendalltau(self.data[column], self.data[target_name])
                #for now, i am ignoring the pvalues
                if not np.isnan(coefficient):
                    #if 'nan', either the column or the target_name is constant. using zero in this case
                    correlations[column] = coefficient
                else:
                    correlations[column] = 0.0
        return correlations

    def get_covariances_with_target(self, target_name):
        """For every numerical column of the dataset, this method computes its covariance  
        with respect to the target column
        """        
        covariances = {}
        for column in self.data:
            if column != target_name and self._is_numerical(column) and self._is_numerical(target_name):
                covariance = cov(self.data[column], self.data[target_name])[0,1]
                if not np.isnan(covariance):
                    covariances[column] = covariance
        return covariances

    def get_normalized_mutual_information_with_target(self, target_name):
        """For every numerical column of the dataset, this method computes its normalized mutual 
        information with respect to the target column
        """        
        mutual_infos = {}
        for column in self.data:
            if column != target_name and self._is_numerical(column) and self._is_numerical(target_name):
                norm_mutual_info = normalized_mutual_info_score(self.data[column], self.data[target_name])
                mutual_infos[column] = norm_mutual_info
        return mutual_infos

    def get_pairwise_features_with_target(self, target_name, func=max):
        """Computes all features of the dataset that concern numerical columns and the target column, 
        returning for example the maximum (in modulus) of them --- it depends on parameter func
        """        
        features = []
        pearson = func(self.get_pearson_correlations_with_target(target_name).values())
        spearman = func(self.get_spearman_correlations_with_target(target_name).values())
        kendalltau = func(self.get_kendall_tau_correlations_with_target(target_name).values())
        covariance = func(self.get_covariances_with_target(target_name).values())
        mutual_info = func(self.get_normalized_mutual_information_with_target(target_name).values())
        #TODO refactor to avoid repetition
        features.append(pearson) if not np.isnan(pearson) else features.append(0.0)
        features.append(spearman) if not np.isnan(spearman) else features.append(0.0)
        features.append(kendalltau) if not np.isnan(kendalltau) else features.append(0.0)
        features.append(covariance) if not np.isnan(covariance) else features.append(0.0)
        features.append(mutual_info) if not np.isnan(mutual_info) else features.append(0.0)
        return features

    def compute_percentual_difference_in_number_of_rows(self, keys_of_another_dataset):
        """Computes the ratio between the number of rows of the dataset and the number of rows of 
        another dataset, using keys and their intersections as a proxy
        """
        final_number_of_rows = len(set(self.data.index.values) & set(keys_of_another_dataset))  
        return final_number_of_rows/len(keys_of_another_dataset)
