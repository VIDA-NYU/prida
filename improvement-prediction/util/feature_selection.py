from sklearn.feature_selection import f_regression, mutual_info_regression
import numpy as np
                 
def f_test_univariate_selection(X, y):
    """Given data instances (X) and their corresponding targets (y),
    this method indicates which features of X are most correlated with y
    according to an f_test based univariate feature selection
    """
    f_test, _ = f_regression(X, y)
    f_test /= np.max(f_test)
    print('ranked features -- f-test', f_test)


def mutual_information_univariate_selection(X, y):
    """Given data instances (X) and their corresponding targets (y),
    this method indicates which features of X are most correlated with y
    according to a mutual information based univariate feature selection
    """
    mi = mutual_info_regression(X, y)
    mi /= np.max(mi)
    dict_mi = {i:j for i, j in enumerate(mi)} 
    print('ranked features -- mutual information', sorted(dict_mi, key=dict_mi.get, reverse=True))
    
