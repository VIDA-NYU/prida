import numpy as np
from dataset import *
from feature_factory import *
from util.metrics import *
from constants import *

class AugmentationInstance:
    def __init__(self, instance_values):
        """This class concerns augmentation instances --- each composed of a query dataset,
        a target variable/column, a candidate dataset for augmentation, and optional metrics regarding 
        the quality of the target prediction before and after augmentation
        """
        self.instance_values_dict = instance_values
        self.query_filename = self.instance_values_dict['query_dataset_id']
        self.query_dataset = Dataset()
        self.query_dataset.initialize_from_data(
            self.instance_values_dict['query_data'],
            key=self.instance_values_dict['query_key']
        )
        self.candidate_filename = self.instance_values_dict['candidate_dataset_id']
        self.candidate_dataset = Dataset()
        self.candidate_dataset.initialize_from_data(
            self.instance_values_dict['candidate_data'],
            key=self.instance_values_dict['candidate_key']
        )
        self.target_name = self.instance_values_dict['target_name']
        self.imputation_strategy = self.instance_values_dict['imputation_strategy']
        self.mark = self.instance_values_dict['mark']
            
        # if the augmentation instance does not come from the test data, the prediction metrics before and
        # after augmentation are available
        if 'mae_before' in instance_values:
            self.mae_before = self.instance_values_dict['mae_before']
            self.mae_after = self.instance_values_dict['mae_after']
            self.mse_before = self.instance_values_dict['mse_before']
            self.mse_after = self.instance_values_dict['mse_after']
            self.med_ae_before = self.instance_values_dict['med_ae_before']
            self.med_ae_after = self.instance_values_dict['med_ae_after']
            self.r2_score_before = self.instance_values_dict['r2_score_before']
            self.r2_score_after = self.instance_values_dict['r2_score_after']

        # if the augmentation instance needs to be composed with a candidate dataset, and the prediction metrics
        # need to be computed with a learning model, such metrics are not present
        else:
            self.mae_before = np.nan
            self.mae_after = np.nan
            self.mse_before = np.nan
            self.mse_after = np.nan
            self.med_ae_before = np.nan
            self.med_ae_after = np.nan
            self.r2_score_before = np.nan
            self.r2_score_after = np.nan

        if self.instance_values_dict['joined_dataset']:
            self.joined_dataset = Dataset()
            self.joined_dataset.initialize_from_filename(
                self.instance_values_dict['joined_dataset'],
                key=self.instance_values_dict['query_key']
            )
            self.containment_score = self.get_key_ratio_between_query_and_candidate_datasets(use_join=True)
            self.joined_dataset.handle_missing_values(self.imputation_strategy)
        else:
            self.joined_dataset = self.join_query_and_candidate_datasets()
            self.containment_score = self.get_key_ratio_between_query_and_candidate_datasets()
        
    def get_query_dataset(self):
        """Returns the query dataset of the augmentation instance (Dataset class)
        """
        return self.query_dataset

    def get_candidate_dataset(self):
        """Returns the candidate dataset of the augmentation instance (Dataset class)
        """
        return self.candidate_dataset

    def get_query_filename(self):
        """ Returns the filename of the query data
        """
        return self.query_filename

    def get_candidate_filename(self):
        """Returns the filename of the candidate data
        """
        return self.candidate_filename

    def get_formatted_fields(self):
        """Returns all instance values (fields) formatted as a dict
        """
        return self.instance_values_dict

    def get_mark(self):
        """Returns whether the data point is a positive example or a negative example.
        Returns None if such information is not available.
        """
        return self.mark
             
    def join_query_and_candidate_datasets(self):
        """Creates a new dataset (Dataset class) by performing a join 
        between the query and candidate datasets. By default, the joining key is 
        'key-for-ranking'
        """
        result_data = self.query_dataset.join_with(self.candidate_dataset, missing_value_imputation=self.imputation_strategy)
        dataset = Dataset()
        dataset.initialize_from_data_and_column_names(result_data)
        return dataset

    def get_joined_query_data(self):
        """Returns the query columns from the joined query+candidate dataset
        """
        query_column_names = self.query_dataset.get_column_names()
        return self.joined_dataset.get_data_columns(query_column_names, '_l')

    def get_joined_candidate_data(self):
        """Returns the candidate columns from the joined query+candidate dataset
        """
        candidate_column_names = self.candidate_dataset.get_column_names()
        return self.joined_dataset.get_data_columns(candidate_column_names, '_r')

    def get_joined_candidate_data_and_target(self):
        """Returns the candidate columns, and the target column, from the joined query+candidate dataset
        """
        column_names = self.candidate_dataset.get_column_names().tolist() + [self.target_name] 
        return self.joined_dataset.get_data_columns(column_names, '_r')

    def get_joined_data(self):
        """Returns the joined query+candidate dataset
        """
        return self.joined_dataset.get_data()

    def get_target_name(self):
        """Returns the name of the target column, which belongs originally to the
        query dataset
        """
        return self.target_name

    # TODO maybe move method below back to feature factory
    def get_key_ratio_between_query_and_candidate_datasets(self, use_join=False):
        """Computes the ratio between the keys that are common for both query and candidate 
        datasets over all keys in the query dataset
        """
        if use_join:
            common_columns = list(set(self.joined_dataset.get_data().columns).intersection(set(self.candidate_dataset.get_data().columns)))
            query_size = self.joined_dataset.get_data().shape[0]
            intersection_size = self.joined_dataset.get_data()[self.joined_dataset.get_data()[common_columns].notna().any(1)].shape[0]
            return intersection_size/query_size
        query_data_keys = self.query_dataset.get_keys()
        candidate_data_keys = self.candidate_dataset.get_keys()
        intersection_size = len(query_data_keys & candidate_data_keys)
        return intersection_size/len(query_data_keys)
      
    def compute_decrease_in_mean_absolute_error(self):
        """Returns the relative decrease in mean_absolute_error if mae_before and
        mae_after are defined
        """
        if self.mae_before and self.mae_after:
            return compute_mae_decrease(self.mae_before, self.mae_after)
        return np.nan

    def compute_decrease_in_mean_squared_error(self):
        """Returns the relative decrease in mean_squared_error if mse_before and
        mse_after are defined
        """
        if self.mse_before and self.mse_after:
            return compute_mse_decrease(self.mse_before, self.mse_after)
        return np.nan

    def compute_decrease_in_median_absolute_error(self):
        """Returns the relative decrease in median_absolute_error if med_ae_before and
        med_ae_after are defined
        """
        if self.med_ae_before and self.med_ae_after:
            return compute_med_ae_decrease(self.med_ae_before, self.med_ae_after)
        return np.nan    
            
    def compute_gain_in_r2_score(self):
        """Returns the relative gain in the r2 score if r2_score_before and 
        r2_score_after are defined
        """
        if self.r2_score_before and self.r2_score_after:
            return compute_r2_gain(self.r2_score_before, self.r2_score_after)
        return np.nan

    def get_before_and_after_r2_scores(self):
        """Returns r2_score_before and r2_score_after
        if they are defined
        """
        if self.r2_score_before and self.r2_score_after:
            return self.r2_score_before, self.r2_score_after
        return np.nan, np.nan
    
    def compute_pairwise_features(self):
        """Given the joined query+candidate dataset, this method computes pairwise features considering:
        (1) - the joined query+candidate dataset, considering just the candidate columns and the target
        (2) - the joined query+candidate dataset, considering just the query columns and the target
        (3) - the difference between the max_in_modulus pearson correlation considering query columns and target, and candidate columns and target
        (4) - the number of keys in the intersection of query+candidate dataset over the number of keys of the original query dataset
        """

        # computing (1)
        feature_factory_candidate_with_target = FeatureFactory(self.get_joined_candidate_data_and_target())
        candidate_features_with_target = feature_factory_candidate_with_target.get_pairwise_features_with_target(self.target_name,
                                                                                                               func=max_in_modulus)
        
        # computing (2)
        feature_factory_query = FeatureFactory(self.get_joined_query_data())
        query_features_with_target = feature_factory_query.get_pairwise_features_with_target(self.target_name,
                                                                                           func=max_in_modulus)

        # computing (3)
        pearson_difference_wrt_target = feature_factory_candidate_with_target.compute_difference_in_pearsons_wrt_target(feature_factory_query.get_max_pearson_wrt_target(self.target_name), self.target_name)

        # computing (4)
        # key_ratio = self.get_key_ratio_between_query_and_candidate_datasets()
        key_ratio = self.containment_score
        
        return query_features_with_target + candidate_features_with_target + [pearson_difference_wrt_target] + [key_ratio]
        
    def generate_features(self, query_dataset_individual_features=[], candidate_dataset_individual_features=[]):
        """This method generates features derived from the datasets of the augmentation instance. 
        The recommendation module computes individual features for the query and candidate datasets in order to 
        avoid repeated computations, optimizing the process. This is why the parameters query_individual_dataset_features and 
        candidate_dataset_individual_features can be different from []
        """
        if not query_dataset_individual_features:
            feature_factory_query = FeatureFactory(self.get_query_dataset().get_data())
            query_dataset_individual_features = feature_factory_query.get_individual_features(func=max_in_modulus)
        if not candidate_dataset_individual_features:
            feature_factory_candidate = FeatureFactory(self.get_candidate_dataset().get_data())
            candidate_dataset_individual_features = feature_factory_candidate.get_individual_features(func=max_in_modulus)

        pairwise_features =  self.compute_pairwise_features()
        return np.array(query_dataset_individual_features + candidate_dataset_individual_features + pairwise_features)
