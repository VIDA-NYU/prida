""" Given (1) a model trained for a target X (saved with pickle), 
          (2) the name of the column for target X,
          (3) a test file with compatible features and target X

          this script explores different ways of performing regression and using the results 
          to rank and recommend useful datasets for augmentation. 
"""

import sys
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import pearsonr, kendalltau
import numpy as np
import pickle

METAFEATURES = ['query', 'target', 'candidate']
KEY_SEPARATOR = '*'

def generate_estimates(training_model_filename, target_column_name, test_filename):
  """This function gets  a random forest regressor and generates 
  estimates for the target_column_name in the test data
  """

  test_data = pd.read_csv(test_filename)
  y_test = test_data[target_column_name]
  X_test = test_data.drop([target_column_name] + METAFEATURES, axis=1)

  model = pickle.load(open(training_model_filename, 'rb'))
  test_data['estimate'] = model.predict(X_test)
  print('R2 score', model.score(X_test, y_test))
  return test_data

def parse_rows(dataset_with_predictions, target_column_name):
  """This function extracts different features for combinations of 
  query, target, and candidate
  """
  print('DATASET', dataset_with_predictions)
  candidates_per_query_target = {str(row['query']) + KEY_SEPARATOR + str(row['target']): {} for index, row in dataset_with_predictions.iterrows()}
  for index, row in dataset_with_predictions.iterrows():
    key = str(row['query']) + KEY_SEPARATOR + str(row['target'])
    candidates_per_query_target[key][row['candidate']] = {target_column_name: row[target_column_name], 'estimate': row['estimate']}
  return candidates_per_query_target


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

def rank_candidates_per_query(test_with_estimates, target_column_name):
  """This function gets all candidates for each (query, target) tuple, and ranks them depending 
  on the regressor estimates.
  """
  candidates_per_query_target = parse_rows(test_with_estimates, target_column_name)
  ndcgs = []
  avg_precs = []
  ndcgs_containment = []
  avg_precs_containment = []
  numbers_of_retrieved_candidates = []
  number_of_instances = 0
  for key in candidates_per_query_target.keys():
    query, target = key.split(KEY_SEPARATOR)
    instances = test_with_estimates.loc[(test_with_estimates['query'] == query) & (test_with_estimates['target'] == target) & (test_with_estimates[target_column_name] > 0)]
    number_of_instances += instances.shape[0]  
    try:
      estimated_gains = instances[['candidate', 'estimate']].values.tolist()
      real_gains = instances[['candidate', target_column_name]].values.tolist()
      containment_baseline = instances[['candidate', 'containment_fraction']].values.tolist()
      avg_precs.append(compute_average_precision(real_gains, estimated_gains))
      ndcgs.append(compute_ndcg_at_k(real_gains, estimated_gains))
      avg_precs_containment.append(compute_average_precision(real_gains, containment_baseline))
      ndcgs_containment.append(compute_ndcg_at_k(real_gains,containment_baseline))
      numbers_of_retrieved_candidates.append(instances.shape[0])
    except ValueError:
      continue
  print('total number of instances', number_of_instances)
  print('average number of candidates per query-target (predicted as positive)', np.mean(numbers_of_retrieved_candidates))
  print('MAP:', np.mean(avg_precs))
  print('NDCGs:', np.mean(ndcgs))
  print('MAP - Containment baseline:', np.mean(avg_precs_containment))
  print('NDCGs - Containment baseline:', np.mean(ndcgs_containment))

if __name__ == '__main__':
  training_model_filename = sys.argv[1]
  target_column_name = sys.argv[2]
  test_filename = sys.argv[3]
  
  
  test_with_estimates = generate_estimates(training_model_filename, target_column_name, test_filename)
  rank_candidates_per_query(test_with_estimates, target_column_name)
  #analyze_predictions(test_with_predictions, alpha)
  #regressor = build_regressor_for_ranking_positive_class(training_data, alpha, features)
  #rank_candidates_classified_as_positive(test_with_predictions, regressor, features)
    
