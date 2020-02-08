""" Given (1) two files (one with training and one with test data; both with headers), 
          (2) a threshold alpha above which a gain in R2 squared should correspond to class GOOD GAIN, and 
          (3) a file with the features that should be used for learning, 

          this script explores different ways of combining classification results with other sources of info 
          in order to recommend useful datasets for augmentation. 
"""

import sys
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils import shuffle
from sklearn.metrics import f1_score, classification_report
from scipy.stats import pearsonr, kendalltau
import numpy as np


TARGET_COLUMN = 'gain_in_r2_score'
POSITIVE_CLASS = 'good_gain'
NEGATIVE_CLASS = 'loss'
KEY_SEPARATOR = '*'

def downsample_data(dataset):
  """This function downsamples the number of instances of a class that is over-represented in the dataset.
  It's important to keep the learning 'fair'
  """
  negative =  dataset.loc[dataset['class'] == NEGATIVE_CLASS]
  positive = dataset.loc[dataset['class'] == POSITIVE_CLASS]
  
  sample_size = min([negative.shape[0], positive.shape[0]])
  negative = negative.sample(n=sample_size, random_state=42)
  positive = positive.sample(n=sample_size, random_state=42)
  
  frames = [negative, positive]
  return shuffle(pd.concat(frames), random_state=0)

def determine_classes_based_on_gain_in_r2_score(dataset, alpha, downsample=False):
  """This function determines the class of each row in the dataset based on the value 
  of TARGET_COLUMN
  """
  gains = dataset[TARGET_COLUMN]
  classes = [POSITIVE_CLASS if i > alpha else NEGATIVE_CLASS for i in gains]
  dataset['class'] = classes
  if downsample:
    return downsample_data(dataset)
  return dataset

def generate_predictions(training, test, alpha, features):
  """This function creates a random forest classifier and generates 
  predictions for the test data
  """
  training = determine_classes_based_on_gain_in_r2_score(training, alpha)
  test = determine_classes_based_on_gain_in_r2_score(test, alpha)
  X_train = training[features]
  y_train = training['class']
  X_test = test[features]
  y_test = test['class']

  clf = RandomForestClassifier(random_state=42)
  clf.fit(X_train, y_train)  
  test['pred'] = clf.predict(X_test)
  print(classification_report(y_test, test['pred']))
  test['prob_positive_class'] = [i[0] for i in clf.predict_proba(X_test)]
  return test

def parse_rows(dataset_with_predictions):
  """This function extracts different features for combinations of 
  query, target, and candidate
  """
  candidates_per_query_target = {str(row['query']) + KEY_SEPARATOR + str(row['target']): {} for index, row in dataset_with_predictions.iterrows()}
  for index, row in dataset_with_predictions.iterrows():
    key = str(row['query']) + KEY_SEPARATOR + str(row['target'])
    candidates_per_query_target[key][row['candidate']] = {TARGET_COLUMN: row[TARGET_COLUMN], 'class': row['class'], 'pred': row['pred'], 'pred_prob': row['prob_positive_class']}
  return candidates_per_query_target

def compute_correlation_prob_class_target(candidates_per_query_target):
  """This function computes the overall correlation between the probability of being in 
  the positive class and the value of the target column
  """
  probs_per_query_target = []
  gains_per_query_target = []
  for key in candidates_per_query_target.keys():
    candidates = candidates_per_query_target[key].keys()
    tmp_probs = [candidates_per_query_target[key][candidate]['pred_prob'] for candidate in candidates]
    tmp_gains = [candidates_per_query_target[key][candidate][TARGET_COLUMN] for candidate in candidates]
    probs_per_query_target += tmp_probs
    gains_per_query_target += tmp_gains
  return pearsonr(probs_per_query_target, gains_per_query_target)

def compute_precision_per_query_target(candidates_per_query_target, alpha):
  """This function computes the precision for the positive class for each query-target
  """
  precs = []
  for key in candidates_per_query_target.keys():
    candidates = candidates_per_query_target[key].keys()
    predicted_positive = 0
    real_positive = 0
    for candidate in candidates:
      if candidates_per_query_target[key][candidate]['class'] == POSITIVE_CLASS:
        real_positive += 1
        if candidates_per_query_target[key][candidate]['pred'] == POSITIVE_CLASS:
          predicted_positive += 1
    if real_positive:
      precs.append(predicted_positive/real_positive)
  return precs

def compute_recall_for_top_k_candidates(candidates_per_query_target, alpha, k):
  """This function computes how many of the top k candidates we efficiently retrieve
  """
  top_recall = []
  num_cands = []
  keys_with_at_least_k_relevant_gains = 0
  for key in candidates_per_query_target.keys():
    candidates = candidates_per_query_target[key].keys()
    num_cands.append(len(candidates))
    gains = []
    for candidate in candidates:
      gains.append((candidates_per_query_target[key][candidate][TARGET_COLUMN], candidates_per_query_target[key][candidate]['pred']))
    relevant_gains = [i for i in sorted(gains)[-k:] if i[0] > alpha]
    positive_right = 0
    for (gain, class_) in relevant_gains:
      if class_ == POSITIVE_CLASS:
        positive_right += 1
    if len(relevant_gains) >= k:
      top_recall.append(positive_right/k)
      keys_with_at_least_k_relevant_gains += 1
  print('this recall was computed taking', keys_with_at_least_k_relevant_gains, 'keys out of', len(candidates_per_query_target.keys()), 'into account')
  print('avg and median num of candidates per query-target pair', np.mean(num_cands), np.median(num_cands))
  return top_recall
    
def analyze_predictions(test_with_preds, alpha):
  """This function separates all candidates for each 
  query-target pair and then analyzes how well the classification worked in 
  each case
  """
  candidates_per_query_target = parse_rows(test_with_preds)
  print('correlation between the probability of being in the positive class and the actual gains', compute_correlation_prob_class_target(candidates_per_query_target))
  print('What is the average precision for positive class per query-target?', np.mean(compute_precision_per_query_target(candidates_per_query_target, alpha)))
  print('What is the average recall for the top-5 candidates?', np.mean(compute_recall_for_top_k_candidates(candidates_per_query_target, alpha, 5)))
  print('What is the average recall for the top-1 candidates?', np.mean(compute_recall_for_top_k_candidates(candidates_per_query_target, alpha, 1)))
  print('What is the average recall for the top-3 candidates?', np.mean(compute_recall_for_top_k_candidates(candidates_per_query_target, alpha, 3)))


def build_regressor_for_ranking_positive_class(dataset, alpha, features, regression_target=TARGET_COLUMN):
  """This function builds a regressor based exclusively on positive class' 
  examples present in the dataset
  """
  if regression_target in features:
    print('The target for the regression task cannot be one of the features')
    return
  
  positive_examples = dataset.loc[dataset[TARGET_COLUMN] > alpha]
  X = positive_examples[features]
  y = positive_examples[regression_target]
  regressor = RandomForestRegressor(random_state=42)
  regressor.fit(X, y)
  return regressor

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

def rank_candidates_classified_as_positive(test_with_preds, regressor, features):
  """This function gets all candidates for each (query, target) tuple, selects those that were classified as positive, and 
  uses a regressor to rank them.
  """
  candidates_per_query_target = parse_rows(test_with_preds)
  ndcgs = []
  avg_precs = []
  ndcgs_containment = []
  avg_precs_containment = []
  numbers_of_retrieved_candidates = []
  for key in candidates_per_query_target.keys():
    query, target = key.split(KEY_SEPARATOR)
    instances = test_with_preds.loc[(test_with_preds['query'] == query) & (test_with_preds['target'] == target) & (test_with_preds['pred'] == POSITIVE_CLASS)]
    #print('predictions', [(candidate, predicted_gain) for candidate, predicted_gain in zip(instances['candidate'], regressor.predict(instances[features]))])
    #print('actual gains', instances[['candidate', TARGET_COLUMN]].values.tolist())
    #TODO compare predicted gains with ALL positive actual gains ('class' == POSITIVE_CLASS instead of 'pred' == POSITIVE_CLASS)
    try:
      predicted_gains = [(candidate, predicted_gain) for candidate, predicted_gain in zip(instances['candidate'], regressor.predict(instances[features]))]
      real_gains = instances[['candidate', TARGET_COLUMN]].values.tolist()
      containment_baseline = instances[['candidate', 'containment_fraction']].values.tolist()
      avg_precs.append(compute_average_precision(real_gains, predicted_gains))
      ndcgs.append(compute_ndcg_at_k(real_gains, predicted_gains))
      avg_precs_containment.append(compute_average_precision(real_gains, containment_baseline))
      ndcgs_containment.append(compute_ndcg_at_k(real_gains,containment_baseline))
      numbers_of_retrieved_candidates.append(instances.shape[0])
    except ValueError:
      continue
    #break
  print('average number of candidates per query-target (predicted as positive)', np.mean(numbers_of_retrieved_candidates))
  print('MAP:', np.mean(avg_precs))
  print('NDCGs:', np.mean(ndcgs))
  print('MAP - Containment baseline:', np.mean(avg_precs_containment))
  print('NDCGs - Containment baseline:', np.mean(ndcgs_containment))
  
    # gains = []
    # for candidate in candidates:
    #   gains.append((candidates_per_query_target[key][candidate][TARGET_COLUMN], candidates_per_query_target[key][candidate]['pred']))
    # relevant_gains = [i for i in sorted(gains)[-k:] if i[0] > alpha]
  
if __name__ == '__main__':
  training_filename = sys.argv[1]
  test_filename = sys.argv[2]
  alpha = float(sys.argv[3])
  features = eval(open(sys.argv[4]).readline())
  
  training_data = pd.read_csv(training_filename)
  test_data = pd.read_csv(test_filename)
  test_with_predictions = generate_predictions(training_data, test_data, alpha, features)
  regressor = build_regressor_for_ranking_positive_class(training_data, alpha, features)
  rank_candidates_classified_as_positive(test_with_predictions, regressor, features)
  
  #analyze_predictions(test_with_predictions, alpha)
