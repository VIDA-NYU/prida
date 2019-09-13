import pandas as pd

class Dataset:
    def __init__(self, filename):
        self.filename = filename
        self.read_dataset()

    def read_dataset(self):
        self.data = pd.read_csv(self.filename)
        self.column_names = self.data.columns.to_list()

    def get_data(self):
        return self.data
