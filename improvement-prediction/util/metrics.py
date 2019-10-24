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
    
def compute_r2_gain(r2_score_before, r2_score_after):
    return (r2_score_after - r2_score_before)/np.fabs(r2_score_before)

def compute_ndcg_at_k(real_gains, predicted_gains, k=5):
    real_ranking = sorted(real_gains, key = lambda x:x[1], reverse=True)
    real_relevances = {tuple[0]:len(real_ranking)-index for index, tuple in enumerate(real_ranking)}
    predicted_ranking = sorted(predicted_gains, key = lambda x:x[1], reverse=True)
    predicted_relevances = [real_relevances[i[0]] for i in predicted_ranking][:k]
    ranked_relevances = sorted(predicted_relevances, reverse=True)

    numerator = np.sum(np.asfarray(predicted_relevances)/np.log2(np.arange(2, np.asfarray(predicted_relevances).size + 2)))
    denominator = np.sum(np.asfarray(ranked_relevances)/np.log2(np.arange(2, np.asfarray(ranked_relevances).size + 2)))
    return numerator/denominator 
