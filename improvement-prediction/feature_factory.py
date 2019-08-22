class FeatureFactory:
    def get_dataset_features(self, data):
        print('extraction of features starts here')
        print(data.dtypes)
        #TODO considering ints and floats alone, compute uni-wise features such as
        #number of attributes (columns) and instances (rows), and the fraction of attributes
        #to instances. also, metalearning: https://arxiv.org/pdf/1810.03548.pdf
