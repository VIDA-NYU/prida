import pandas as pd

class Dataset:
    def __init__(self, *args):
        if len(args) == 1:
            filename = args[0]
            self.initialize_from_filename(filename)
        elif len(args) == 2:
            data = args[0]
            column_names = args[1]
            self.initialize_from_data_and_column_names(data, column_names)

    def initialize_from_filename(self, filename):
        self.filename = filename
        self.read_dataset()

    def initialize_from_data_and_column_names(self, data, columns):
        self.data = data
        self.column_names = columns
        
    def read_dataset(self):
        self.data = pd.read_csv(self.filename)
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
