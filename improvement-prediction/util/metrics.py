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
    #print('real gains', real_gains, 'predicted gains', predicted_gains)
    #print('real relevances', real_relevances, 'predicted relevances', predicted_ranking)
    real_relevances_of_predicted_items = [real_relevances[i[0]] if i[0] in real_relevances else 0 for i in predicted_ranking]
    #print('real_relevances_of_predicted_items', real_relevances_of_predicted_items)
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

def compute_r_precision(real_gains, predicted_gains, k=5, positive_only=False):
    """This function computes R-precision, which is the ratio between all the relevant documents 
    retrieved until the rank that equals the number of relevant documents you have in your collection in total (r), 
    to the total number of relevant documents in your collection R.
    
    In this setting, if positive_only == True, relevant documents correspond exclusively to candidates associated to positive 
    gains. If there are no relevant documents in this case, this function returns 'nan'. Alternatively, if positive_only == False 
    we consider that the relevant documents are the k highest ranked candidates in real_gains (it basically turns into precision at k).
    """

    if positive_only:
        relevant_documents = [elem[0] for elem in real_gains if elem[1] > 0]
        predicted_ranking = [elem[0] for elem in sorted(predicted_gains, key = lambda x:x[1], reverse=True)[:len(relevant_documents)]]
        if not relevant_documents or not predicted_ranking:
            return float('nan')
        return len(set(relevant_documents) & set(predicted_ranking))/len(relevant_documents)

    #positive_only == False
    real_ranking = [elem[0] for elem in sorted(real_gains, key = lambda x:x[1], reverse=True)[:k]]
    predicted_ranking = [elem[0] for elem in sorted(predicted_gains, key = lambda x:x[1], reverse=True)[:k]]
    return len(set(real_ranking) & set(predicted_ranking))/k

def compute_average_precision(real_gains, predicted_gains):
    """This function computes average precision, which is the average of precision at k values for k=1..len(real_gains).
    Average precision values can later be used for the computation of MAP (Mean Average Precision)
    """
    precs = [compute_r_precision(real_gains, predicted_gains, k=x+1) for x in range(len(real_gains))]
    if not precs:
        return 0.
    return np.mean(precs)
