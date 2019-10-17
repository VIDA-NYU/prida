from sklearn.metrics import mean_squared_error
import numpy as np

def compute_MSE(predicted_values, real_values):
    return mean_squared_error(predicted_values, real_values)

def compute_squared_error_distribution(predicted_values, real_values, normalize=False):
    distribution = [(p - r)**2 for p, r in zip(predicted_values, real_values)]
    if not normalize:
        return distribution

    denominator = max(distribution)
    return [i/denominator for i in distribution]

def compute_SMAPE(predicted_values, real_values):
    return 100/len(real_values) * np.sum(2 * np.fabs(predicted_values - real_values) / (np.fabs(real_values) + np.fabs(predicted_values)))

def compute_symmetric_absolute_percentage_error_distribution(predicted_values, real_values, normalize=False):
    distribution = [2*np.fabs(p - r)/(np.fabs(r) + np.fabs(p)) for p, r in zip(predicted_values, real_values)]
    if not normalize:
        return distribution

    denominator = max(distribution)
    return [i/denominator for i in distribution]
    

