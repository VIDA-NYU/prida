import numpy as np
from dataset import *
from feature_factory import *
from util.metrics import *

class AugmentationInstance:
    def __init__(self, instance_values, use_hdfs=False, hdfs_address=None, hdfs_user=None):
        """This class concerns augmentation instances --- each composed of a query dataset,
        a target variable/column, a candidate dataset for augmentation, and optional metrics regarding 
        the quality of the target prediction before and after augmentation
        """
        self.query_filename = instance_values['query_filename']
        self.query_dataset = Dataset()
        self.query_dataset.initialize_from_filename(self.query_filename, use_hdfs, hdfs_address, hdfs_user)
        self.candidate_filename = instance_values['candidate_filename']
        self.candidate_dataset = Dataset()
        self.candidate_dataset.initialize_from_filename(self.candidate_filename, use_hdfs, hdfs_address, hdfs_user)
        self.target_name = instance_values['target_name']

        # if the augmentation instance comes from the training data, the prediction metrics before and
        # after augmentation are available
        if len(instance_values.keys()) == 5:
            self.r2_score_before = instance_values['r2_score_before']
            self.r2_score_after = instance_values['r2_score_after']

        # if the augmentation instance needs to be composed with a candidate dataset, and the prediction metrics
        # need to be computed with a learning model, such metrics are not present
        elif len(instance_values.keys()) == 3:
            self.r2_score_before = np.nan
            self.r2_score_after = np.nan
        self.joined_dataset = self.join_query_and_candidate_datasets()
        
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
        fields = {'query_filename': self.query_filename,
                  'target_name': self.target_name,
                  'candidate_filename': self.candidate_filename,
                  'r2_score_before': self.r2_score_before,
                  'r2_score_after': self.r2_score_after}
        return fields
             
    def join_query_and_candidate_datasets(self):
        """Creates a new dataset (Dataset class) by performing an inner join 
        between the query and candidate datasets. By default, the joining key is 
        'key-for-ranking'
        """
        result_data = self.query_dataset.join_with(self.candidate_dataset, key='key-for-ranking')
        dataset = Dataset()
        dataset.initialize_from_data_and_column_names(result_data, result_data.columns)
        return dataset

    def get_joined_query_data(self):
        """Returns the query columns from the joined query+candidate dataset
        """
        query_column_names = self.query_dataset.get_column_names()
        return self.joined_dataset.get_data_columns(query_column_names, '_left')

    def get_joined_candidate_data(self):
        """Returns the candidate columns from the joined query+candidate dataset
        """
        candidate_column_names = self.candidate_dataset.get_column_names()
        return self.joined_dataset.get_data_columns(candidate_column_names, '_right')

    def get_joined_candidate_data_and_target(self):
        """Returns the candidate columns, and the target column, from the joined query+candidate dataset
        """
        column_names = self.candidate_dataset.get_column_names().tolist() + [self.target_name] 
        return self.joined_dataset.get_data_columns(column_names, '_right')

    def get_joined_data(self):
        """Returns the joined query+candidate dataset
        """
        return self.joined_dataset.get_data()

    def get_target_name(self):
        """Returns the name of the target column, which belongs originally to the
        query dataset
        """
        return self.target_name

    def compute_gain_in_r2_score(self):
        """Returns the relative gain in the r2 score if r2_score_before and 
        r2_score_after are defined
        """
        if self.r2_score_before and self.r2_score_after:
            return compute_r2_gain(self.r2_score_before, self.r2_score_after)
        return np.nan

    def compute_pairwise_features(self):
        """Given the joined query+candidate dataset, this method computes pairwise features considering:
        (1) - the joined query+candidate dataset
        (2) - the joined query+candidate dataset with respect to the target alone
        (3) - the joined query+candidate dataset, considering just the candidate columns and the target
        (4) - the joined query+candidate dataset, considering just the query columns and the target
        (5) - the difference between the max_in_modulus pearson correlation considering query columns and target, and candidate columns and target
        (6) - the number of rows of the joined query+candidate dataset over the number of rows of the original query dataset
        """

        # computing (1)
        feature_factory_joined_dataset = FeatureFactory(self.get_joined_data())
        joined_dataset_features = feature_factory_joined_dataset.get_pairwise_features(func=max_in_modulus)

        # computing (2)
        joined_dataset_features_with_target = feature_factory_joined_dataset.get_pairwise_features_with_target(self.target_name,
                                                                                                             func=max_in_modulus)

        # computing (3)
        feature_factory_candidate_with_target = FeatureFactory(self.get_joined_candidate_data_and_target())
        candidate_features_with_target = feature_factory_candidate_with_target.get_pairwise_features_with_target(self.target_name,
                                                                                                               func=max_in_modulus)
        
        # computing (4)
        feature_factory_query = FeatureFactory(self.get_joined_query_data())
        query_features_with_target = feature_factory_query.get_pairwise_features_with_target(self.target_name,
                                                                                           func=max_in_modulus)

        # computing (5)
        pearson_difference_wrt_target = feature_factory_candidate_with_target.compute_difference_in_pearsons_wrt_target(feature_factory_query.get_max_pearson_wrt_target(self.target_name),
                                                                                                                        self.target_name)

        # computing (6)
        difference_in_numbers_of_rows = feature_factory_candidate_with_target.compute_percentual_difference_in_number_of_rows(self.query_dataset.get_data().shape[0])
        
        return fd_features + features_with_target + query_features_with_target + candidate_features_with_target + [pearson_difference_wrt_target] + [difference_in_numbers_of_rows]
        
    def generate_features(self, query_dataset_individual_features=[], candidate_dataset_individual_features=[]):
        """This method generates features derived from the datasets of the augmentation instance. 
        The recommendation module computes individual features for the query and candidate datasets in order to 
        avoid repeated computations, optimizing the process. This is why the parameters query_individual_dataset_features and 
        candidate_dataset_individual_features can be different from []
        """
        if not query_individual_features:
            feature_factory_query = FeatureFactory(self.get_joined_query_data())
            query_dataset_individual_features = feature_factory_query.get_individual_features(func=max_in_modulus)
        if not candidate_individual_features:
            feature_factory_candidate = FeatureFactory(self.get_joined_candidate_data())
            candidate_dataset_individual_features = feature_factory_candidate.get_individual_features(func=max_in_modulus)

        pairwise_features = self.compute_pairwise_features()
        return np.array(query_individual_dataset_features + candidate_dataset_individual_features + pairwise_features)
