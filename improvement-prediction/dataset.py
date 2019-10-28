#!/usr/bin/env python3
import pandas as pd
from io import StringIO
from util.file import *

class Dataset:
    """This class stores and manages a certain dataset and performs join operations with other given 
    datasets
    """    
    def initialize_from_filename(self, filename, use_hdfs=False, hdfs_address=None, hdfs_user=None):
        """Generates a pandas representation of a dataset read from a given filename
        """
        self.filename = filename
        self.read_dataset(use_hdfs, hdfs_address, hdfs_user)

    def initialize_from_data_and_column_names(self, data, columns):
        """Stores a pandas representation of a dataset that is passed in the parameters
        """
        self.data = data
        self.column_names = columns
        
    def read_dataset(self, use_hdfs=False, hdfs_address=None, hdfs_user=None):
        """Reads lines from self.filename, storing in a pandas dataframe
        """
        self.data = pd.read_csv(StringIO(read_file(self.filename, use_hdfs, hdfs_address, hdfs_user)))
        self.column_names = self.data.columns

    def get_data(self):
        """Returns the dataset (a pandas dataframe)
        """
        return self.data

    def get_column_names(self):
        """Returns the names of the dataset columns
        """
        return self.column_names
    
    def join_with(self, another_dataset, key='key'):
        """Performs inner join between the dataset and another given dataset. _left and _right suffixes are added to columns to 
        avoid overwriting in the case where both datasets have columns with the same name. Null values are also removed
        """
        return self.data.join(another_dataset.get_data().set_index(key), on=key, how='left', lsuffix='_left', rsuffix='_right').dropna()

    def _determine_correct_column_names(self, column_names, possible_suffix):
        """Given column names, this method returns their corresponding actual names, which can either be the column 
        names themselves or the column names plus a possible_suffix, in case an inner join was previously performed
        """
        correct_column_names = []
        for cn in column_names:
            if cn in self.column_names:
                correct_column_names.append(cn)
            else:
                correct_column_names.append(cn + possible_suffix)
        return correct_column_names

    def get_data_columns(self, column_names, possible_suffix):
        """Returns the column names of the dataset
        """
        correct_column_names = self._determine_correct_column_names(column_names, possible_suffix)
        return self.data[correct_column_names]
