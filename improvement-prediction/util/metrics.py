from sklearn.metrics import mean_squared_error
from scipy.stats import kendalltau
import numpy as np

def compute_MSE(predicted_values, real_values):
    """Given aligned predicted and real values, computes
    the mean squared error between them
    """
    return mean_squared_error(predicted_values, real_values)

def compute_squared_error_distribution(predicted_values, real_values, normalize=False):
    """Given aligned predicted and real values, computes
    the distribution of squared errors between them
    """
    distribution = [(p - r)**2 for p, r in zip(predicted_values, real_values)]
    if not normalize:
        return distribution

    denominator = max(distribution)
    return [i/denominator for i in distribution]

def compute_SMAPE(predicted_values, real_values):
    """Given aligned predicted and real values, computes
    the SMAPE (symmetric mean absolute percentage error) between them
    """
    return 100/len(real_values) * np.sum(2 * np.fabs(predicted_values - real_values) / (np.fabs(real_values) + np.fabs(predicted_values)))

def compute_symmetric_absolute_percentage_error_distribution(predicted_values, real_values, normalize=False):
    """Given aligned predicted and real values, computes
    the distribution of symmetric absolute percentage errors between them
    """
    distribution = [2*np.fabs(p - r)/(np.fabs(r) + np.fabs(p)) for p, r in zip(predicted_values, real_values)]
    if not normalize:
        return distribution

    denominator = max(distribution)
    return [i/denominator for i in distribution]
    
def compute_r2_gain(r2_score_before, r2_score_after):
    """Given two r2 scores, corresponding to the prediction of 
    a target column before and after data augmentation, computes 
    their relative gain
    """
    return (r2_score_after - r2_score_before)/np.fabs(r2_score_before)

def compute_mae_decrease(mae_before, mae_after):
    """Given two mean absolute errors, corresponding to the prediction of 
    a target column before and after data augmentation, computes 
    their relative decrease
    """
    return (mae_after - mae_before)/mae_before

def compute_mse_decrease(mse_before, mse_after):
    """Given two mean squared errors, corresponding to the prediction of 
    a target column before and after data augmentation, computes 
    their relative decrease
    """
    return (mse_after - mse_before)/mse_before

def compute_med_ae_decrease(med_ae_before, med_ae_after):
    """Given two median absolute errors, corresponding to the prediction of 
    a target column before and after data augmentation, computes 
    their relative decrease
    """
    return (med_ae_after - med_ae_before)/med_ae_before
    
def compute_ndcg_at_k(real_gains, predicted_gains, k=5, use_gains_as_relevance_weights=False):
    """Given real gains and predicted gains, computes the ndcg between them for the first k positions of 
    the ranking. The relevance weights can be the real relative gains or numbers indicating their rank 
    positions
    """
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
    """Given real gains and predicted gains, computes the kendall-tau distance between them. 
    The higher the real gain, the higher its position in the ranking
    """
    ranked_candidates = [i[0] for i in sorted(real_gains, key = lambda x:x[1], reverse=True)]
    predicted_candidates = [i[0] for i in sorted(predicted_gains, key = lambda x:x[1], reverse=True)]
    return kendalltau(ranked_candidates, predicted_candidates)

def compute_mean_reciprocal_rank_for_single_sample(real_gains, predicted_gains):
    """Given real gains and predicted gains, computes the mean reciprocal rank (MRR) between them.
    Ideally, this metric should be used over large lists (multiple samples) of real gains and 
    predicted gains
    """
    real_ranking = sorted(real_gains, key = lambda x:x[1], reverse=True)
    real_best_candidate = real_ranking[0][0]
    predicted_ranking = sorted(predicted_gains, key = lambda x:x[1], reverse=True)
    for index, elem in enumerate(predicted_ranking):
        if real_best_candidate == elem[0]:
            return 1/(index + 1)
    return 0

def compute_precision_at_k(real_gains, predicted_gains, k=5):
    """This function computes precision@k, mathematically defined
    as (# of recommended items @k that are relevant) /k

    Given a certain k in this implementation, we assume that an item is 
    relevant if it is one of the top k in real_gains
    """
    real_ranking = dict(sorted(real_gains, key = lambda x:x[1], reverse=True)[:k])
    predicted_ranking = sorted(predicted_gains, key = lambda x:x[1], reverse=True)[:k]
    num = 0.
    den = k
    for item in predicted_ranking:
        if item[0] in real_ranking: 
            num += 1.
    return num/den
