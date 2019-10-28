import numpy as np

def remove_outliers(X, y, zscore_threshold=3):
    """Given data instances (X) and their corresponding targets (y),
    this method removes those instances whose targets are outliers 
    according to the zscore criterion
    """
    mean_ = np.mean(y)
    std_ = np.std(y)
    #print('mean of the data', mean_, 'std of the data', std_)
    X_filtered = []
    y_filtered = []
    for X_value, y_value in zip(X, y):
        zscore = (y_value - mean_)/std_
        if np.fabs(zscore) <= zscore_threshold:
            X_filtered.append(X_value)
            y_filtered.append(y_value)
    return np.array(X_filtered), np.array(y_filtered)

def remove_large_points(X, y, threshold=0.1):
    """Given data instances (X), their corresponding targets (y), 
    and a threshold, this method removes those instances whose targets 
    are larger, in modulus, than the threshold
    """
    X_filtered = []
    y_filtered = []
    for X_value, y_value in zip(X, y):
        if np.fabs(y_value) <= threshold:
            X_filtered.append(X_value)
            y_filtered.append(y_value)
    return np.array(X_filtered), np.array(y_filtered)

def remove_small_points(X, y, threshold=10):
    """Given data instances (X), their corresponding targets (y), 
    and a threshold, this method removes those instances whose targets 
    are smaller, in modulus, than the threshold
    """
    X_filtered = []
    y_filtered = []
    for X_value, y_value in zip(X, y):
        if np.fabs(y_value) > threshold:
            X_filtered.append(X_value)
            y_filtered.append(y_value)
    return np.array(X_filtered), np.array(y_filtered)

