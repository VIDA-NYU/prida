from sklearn.metrics import mean_squared_error
from scipy.stats import kendalltau
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

def compute_ndcg_at_k(real_gains, predicted_gains, k=5, use_gains_as_relevance_weights=False):
    real_ranking = sorted(real_gains, key = lambda x:x[1], reverse=True)[:k]
    if use_gains_as_relevance_weights:
        real_relevances = [(tuple[0], tuple[1]) if tuple[1] > 0 else (tuple[0],0.0) for index, tuple in enumerate(real_ranking)]
        real_relevances = dict(real_relevances)
    else:
        real_relevances = {tuple[0]:len(real_ranking)-index for index, tuple in enumerate(real_ranking)}
    predicted_ranking = sorted(predicted_gains, key = lambda x:x[1], reverse=True)
    real_relevances_of_predicted_items = [real_relevances[i[0]] if i[0] in real_relevances else 0 for i in predicted_ranking]
    print('real_relevances_of_predicted_items', real_relevances_of_predicted_items)
    numerator = np.sum(np.asfarray(real_relevances_of_predicted_items)/np.log2(np.arange(2, np.asfarray(real_relevances_of_predicted_items).size + 2)))
    sorted_real_relevances_of_predicted_items = sorted(real_relevances_of_predicted_items, reverse=True)
    denominator = np.sum(np.asfarray(sorted_real_relevances_of_predicted_items)/np.log2(np.arange(2, np.asfarray(sorted_real_relevances_of_predicted_items).size + 2)))
    return numerator/denominator 

def compute_kendall_tau(real_gains, predicted_gains):
    ranked_candidates = [i[0] for i in sorted(real_gains, key = lambda x:x[1], reverse=True)]
    predicted_candidates = [i[0] for i in sorted(predicted_gains, key = lambda x:x[1], reverse=True)]
    return kendalltau(ranked_candidates, predicted_candidates)

def compute_mean_reciprocal_rank_for_single_sample(real_gains, predicted_gains):
    real_ranking = sorted(real_gains, key = lambda x:x[1], reverse=True)
    real_best_candidate = real_ranking[0][0]
    predicted_ranking = sorted(predicted_gains, key = lambda x:x[1], reverse=True)
    for index, elem in enumerate(predicted_ranking):
        if real_best_candidate == elem[0]:
            return 1/(index + 1)
    return 0

    


