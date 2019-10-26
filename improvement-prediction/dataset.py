#!/usr/bin/env python3
import pandas as pd
from io import StringIO
from util.file import *

class Dataset:
    
    def initialize_from_filename(self, filename, use_hdfs=False, hdfs_address=None, hdfs_user=None):
        self.filename = filename
        self.read_dataset(use_hdfs, hdfs_address, hdfs_user)

    def initialize_from_data_and_column_names(self, data, columns):
        self.data = data
        self.column_names = columns
        
    def read_dataset(self, use_hdfs=False, hdfs_address=None, hdfs_user=None):
        self.data = pd.read_csv(StringIO(read_file(self.filename, use_hdfs, hdfs_address, hdfs_user)))
        self.column_names = self.data.columns

    def get_data(self):
        return self.data

    def get_column_names(self):
        return self.column_names
    
    def join_with(self, another_dataset, key='key'):
        return self.data.join(another_dataset.get_data().set_index(key), on=key, how='left', lsuffix='_left', rsuffix='_right').dropna()

    def _determine_correct_column_names(self, column_names, possible_suffix):
        correct_column_names = []
        for cn in column_names:
            if cn in self.column_names:
                correct_column_names.append(cn)
            else:
                correct_column_names.append(cn + possible_suffix)
        return correct_column_names

    def get_data_columns(self, column_names, possible_suffix):
        correct_column_names = self._determine_correct_column_names(column_names, possible_suffix)
        return self.data[correct_column_names]
