""" Given (1) two files (one with training and one with test data; both with headers), 
          (2) a file with the features that should be used for learning, and
          (3) a compatible regression model, saved with pickle, to be used as a baseline

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
import pickle

TARGET_COLUMN = 'gain_in_r2_score'
POSITIVE_CLASS = 'good_gain'
NEGATIVE_CLASS = 'loss'
KEY_SEPARATOR = '*'
ALPHA = 0

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

def determine_classes_based_on_gain_in_r2_score(dataset, downsample=False):
  """This function determines the class of each row in the dataset based on the value 
  of TARGET_COLUMN
  """
  gains = dataset[TARGET_COLUMN]
  classes = [POSITIVE_CLASS if i > ALPHA else NEGATIVE_CLASS for i in gains]
  dataset['class'] = classes
  if downsample:
    return downsample_data(dataset)
  return dataset

def generate_predictions(training, test, features):
  """This function creates a random forest classifier and generates 
  predictions for the test data
  """
  training = determine_classes_based_on_gain_in_r2_score(training)
  test = determine_classes_based_on_gain_in_r2_score(test)
  X_train = training[features]
  y_train = training['class']
  X_test = test[features]
  y_test = test['class']

  clf = RandomForestClassifier(random_state=42, n_estimators=100)
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

def compute_precision_per_query_target(candidates_per_query_target):
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

def compute_recall_for_top_k_candidates(candidates_per_query_target, k):
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
    relevant_gains = [i for i in sorted(gains)[-k:] if i[0] > ALPHA]
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
    
def analyze_predictions(test_with_preds):
  """This function separates all candidates for each 
  query-target pair and then analyzes how well the classification worked in 
  each case
  """
  candidates_per_query_target = parse_rows(test_with_preds)
  print('correlation between the probability of being in the positive class and the actual gains', compute_correlation_prob_class_target(candidates_per_query_target))
  print('What is the average precision for positive class per query-target?', np.mean(compute_precision_per_query_target(candidates_per_query_target)))
  print('What is the average recall for the top-5 candidates?', np.mean(compute_recall_for_top_k_candidates(candidates_per_query_target, 5)))
  print('What is the average recall for the top-1 candidates?', np.mean(compute_recall_for_top_k_candidates(candidates_per_query_target, 1)))
  print('What is the average recall for the top-3 candidates?', np.mean(compute_recall_for_top_k_candidates(candidates_per_query_target, 3)))


def build_regressor_for_ranking_positive_class(dataset, features, regression_target=TARGET_COLUMN):
  """This function builds a regressor based exclusively on positive class' 
  examples present in the dataset
  """
  if regression_target in features:
    print('The target for the regression task cannot be one of the features')
    return
  
  positive_examples = dataset.loc[dataset[TARGET_COLUMN] > ALPHA]
  X = positive_examples[features]
  y = positive_examples[regression_target]
  regressor = RandomForestRegressor(random_state=20)
  regressor.fit(X, y)
  return regressor

def compute_r_precision(real_gains, predicted_gains, k=5, positive_only=True):
  """This function computes R-precision, which is the ratio between all the relevant documents 
  retrieved until the rank that equals the number of relevant documents you have in your collection in total (r), 
  to the total number of relevant documents in your collection R.
  
  In this setting, if positive_only == True, relevant documents correspond exclusively to candidates associated to real positive 
  gains. If there are no relevant documents in this case, this function returns 'nan'. Alternatively, if positive_only == False 
  we consider that the relevant documents are the k highest ranked candidates in real_gains.
  """
  
  if positive_only:
    relevant_documents = [elem[0] for elem in sorted(real_gains, key = lambda x:x[1], reverse=True) if elem[1] > 0][:k]
    #print('real gains in compute_r_precision', real_gains)
    #print('relevant documents (positive only)', relevant_documents)
    predicted_ranking = [elem[0] for elem in sorted(predicted_gains, key = lambda x:x[1], reverse=True)][:len(relevant_documents)]
    if relevant_documents and predicted_ranking:
      return len(set(relevant_documents) & set(predicted_ranking))/len(relevant_documents)
    return float('nan')
  
  #positive_only == False
  real_ranking = [elem[0] for elem in sorted(real_gains, key = lambda x:x[1], reverse=True)][:k]
  predicted_ranking = [elem[0] for elem in sorted(predicted_gains, key = lambda x:x[1], reverse=True)][:k]
  return len(set(real_ranking) & set(predicted_ranking))/k

def compute_r_recall(real_gains, predicted_gains, k=5, positive_only=True):
  """This function computes 'R-recall' (does it exist officially?), defined as the ratio between all the top-k relevant documents 
  retrieved until the rank k and the total number of retrieved documents (should be k).
  
  In this setting, if positive_only == True, relevant documents correspond exclusively to candidates associated to real positive 
  gains. If there are no relevant documents in this case, this function returns 'nan'. Alternatively, if positive_only == False 
  we consider that the relevant documents are the k highest ranked candidates in real_gains.
  """
  ranking = [elem[0] for elem in sorted(predicted_gains, key = lambda x:x[1], reverse=True)][:k]
  if positive_only:
    top_k_relevant_documents = [elem[0] for elem in sorted(real_gains, key = lambda x:x[1], reverse=True) if elem[1] > 0][:k]
  else:
    top_k_relevant_documents = [elem[0] for elem in sorted(real_gains, key = lambda x:x[1], reverse=True)][:k]
  if top_k_relevant_documents and ranking:
    return len(set(top_k_relevant_documents) & set(ranking))/len(ranking)
  return float('nan')
  
def compute_average_precision(real_gains, predicted_gains):
  """This function computes average precision, which is the average of precision at k values for k=1..len(real_gains).
  Average precision values can later be used for the computation of MAP (Mean Average Precision)
  """
  precs = [compute_r_precision(real_gains, predicted_gains, k=x+1) for x in range(len(real_gains))]
  if not precs:
    return 0.
  return np.mean(precs)

def compute_precision_at_k(real_gains, predicted_gains, k=5):
  """This function computes precision-at-k, i.e. the proportion of recommended items in the top-k set 
  that are relevant (belong to real_gains).

  NOTE THAT IT IS NOT AS STRICT AS compute_r_precision!
  """
  relevant_documents = [elem[0] for elem in real_gains]
  retrieved_documents = [elem[0] for elem in sorted(predicted_gains, key = lambda x:x[1], reverse=True)][:k]
  if relevant_documents and retrieved_documents:
    return len(set(relevant_documents) & set(retrieved_documents))/len(retrieved_documents)
  return float('nan')

def compute_recall_at_k(real_gains, predicted_gains, k=5):
  """This function computes recall-at-k, i.e. the proportion of relevant items found in the top-k 
  recommendations.
  """
  relevant_documents = [elem[0] for elem in real_gains]
  retrieved_documents = [elem[0] for elem in sorted(predicted_gains, key = lambda x:x[1], reverse=True)][:k]
  if relevant_documents and retrieved_documents:
    return len(set(relevant_documents) & set(retrieved_documents))/len(relevant_documents)
  return float('nan')


def rank_candidates_classified_as_positive(test_with_preds, regressor, features, baseline_regressor):
  """This function gets all candidates for each (query, target) tuple, selects those that were classified as positive, and 
  uses a regressor to rank them.
  """
  candidates_per_query_target = parse_rows(test_with_preds)
  avg_precs = []
  avg_precs_certain = []
  avg_precs_containment = []
  avg_precs_baseline_regressor = []
  avg_precs_classif_containment = []
  avg_precs_classif_certain_containment = []
  avg_precs_classif_pearson = []
  avg_precs_classif_certain_pearson = []

  avg_recs = []
  avg_recs_certain = []
  avg_recs_containment = []
  avg_recs_baseline_regressor = []
  avg_recs_classif_containment = []
  avg_recs_classif_certain_containment = []
  avg_recs_classif_pearson = []
  avg_recs_classif_certain_pearson = []
  
  avg_strict_r_precisions = []
  avg_strict_r_precisions_certain = []
  avg_strict_r_precisions_containment = []
  avg_strict_r_precisions_baseline_regressor = []
  avg_strict_r_precisions_classif_containment = []
  avg_strict_r_precisions_classif_certain_containment = []
  avg_strict_r_precisions_classif_pearson = []
  avg_strict_r_precisions_classif_certain_pearson = []

  avg_strict_r_recalls = []
  avg_strict_r_recalls_certain = []
  avg_strict_r_recalls_containment = []
  avg_strict_r_recalls_baseline_regressor = []
  avg_strict_r_recalls_classif_containment = []
  avg_strict_r_recalls_classif_certain_containment = []
  avg_strict_r_recalls_classif_pearson = []
  avg_strict_r_recalls_classif_certain_pearson = []
  
  numbers_of_retrieved_candidates = []
  for key in candidates_per_query_target.keys():
    query, target = key.split(KEY_SEPARATOR)

    instances = test_with_preds.loc[(test_with_preds['query'] == query) & (test_with_preds['target'] == target)]
    truly_positive = instances.loc[test_with_preds['class'] == POSITIVE_CLASS]
    predicted_positive_certain = instances.loc[(test_with_preds['pred'] == POSITIVE_CLASS) & (test_with_preds['prob_positive_class'] > 0.6)]
    predicted_positive = instances.loc[test_with_preds['pred'] == POSITIVE_CLASS]
    if truly_positive.shape[0] and predicted_positive.shape[0] and predicted_positive_certain.shape[0]:
      real_gains = truly_positive[['candidate', TARGET_COLUMN]].values.tolist()
      
      classifier_and_regressor_gains = [[candidate, estimated_gain] for candidate, estimated_gain in zip(predicted_positive['candidate'], regressor.predict(predicted_positive[features]))]
      classifier_certain_and_regressor_gains = [[candidate, estimated_gain] for candidate, estimated_gain in zip(predicted_positive_certain['candidate'], regressor.predict(predicted_positive_certain[features]))]
      
      classifier_and_containment_gains = predicted_positive[['candidate', 'containment_fraction']].values.tolist()
      classifier_certain_and_containment_gains = predicted_positive_certain[['candidate', 'containment_fraction']].values.tolist()

      classifier_and_max_pearson_diff_gains = predicted_positive[['candidate', 'max_pearson_difference']].values.tolist()
      classifier_certain_and_max_pearson_diff_gains = predicted_positive_certain[['candidate', 'max_pearson_difference']].values.tolist()
      
      containment_baseline_gains = instances[['candidate', 'containment_fraction']].values.tolist()
      max_pearson_diff_baseline_gains = instances[['candidate', 'max_pearson_difference']].values.tolist()
      baseline_regressor_gains = [[candidate, baseline_gain] for candidate, baseline_gain in zip(instances['candidate'], regressor.predict(instances[features]))]
      
      avg_precs.append(compute_precision_at_k(real_gains, classifier_and_regressor_gains))
      avg_precs_certain.append(compute_precision_at_k(real_gains, classifier_certain_and_regressor_gains))
      avg_precs_classif_containment.append(compute_precision_at_k(real_gains, classifier_and_containment_gains))
      avg_precs_classif_certain_containment.append(compute_precision_at_k(real_gains, classifier_certain_and_containment_gains))
      avg_precs_classif_pearson.append(compute_precision_at_k(real_gains, classifier_and_max_pearson_diff_gains))
      avg_precs_classif_certain_pearson.append(compute_precision_at_k(real_gains, classifier_certain_and_max_pearson_diff_gains))
      avg_precs_containment.append(compute_precision_at_k(real_gains, containment_baseline_gains))
      avg_precs_baseline_regressor.append(compute_precision_at_k(real_gains, baseline_regressor_gains))
      
      avg_recs.append(compute_recall_at_k(real_gains, classifier_and_regressor_gains))
      avg_recs_certain.append(compute_recall_at_k(real_gains, classifier_certain_and_regressor_gains))
      avg_recs_classif_containment.append(compute_recall_at_k(real_gains, classifier_and_containment_gains))
      avg_recs_classif_certain_containment.append(compute_recall_at_k(real_gains, classifier_certain_and_containment_gains))
      avg_recs_classif_pearson.append(compute_recall_at_k(real_gains, classifier_and_max_pearson_diff_gains))
      avg_recs_classif_certain_pearson.append(compute_recall_at_k(real_gains, classifier_certain_and_max_pearson_diff_gains))
      avg_recs_containment.append(compute_recall_at_k(real_gains, containment_baseline_gains))
      avg_recs_baseline_regressor.append(compute_recall_at_k(real_gains, baseline_regressor_gains))

      avg_strict_r_precisions.append(compute_r_precision(real_gains, classifier_and_regressor_gains))
      avg_strict_r_precisions_certain.append(compute_r_precision(real_gains, classifier_certain_and_regressor_gains))
      avg_strict_r_precisions_classif_containment.append(compute_r_precision(real_gains, classifier_and_containment_gains))
      avg_strict_r_precisions_classif_certain_containment.append(compute_r_precision(real_gains, classifier_certain_and_containment_gains))
      avg_strict_r_precisions_classif_pearson.append(compute_r_precision(real_gains, classifier_and_max_pearson_diff_gains))
      avg_strict_r_precisions_classif_certain_pearson.append(compute_r_precision(real_gains, classifier_certain_and_max_pearson_diff_gains))
      avg_strict_r_precisions_containment.append(compute_r_precision(real_gains, containment_baseline_gains))
      avg_strict_r_precisions_baseline_regressor.append(compute_r_precision(real_gains, baseline_regressor_gains))

      avg_strict_r_recalls.append(compute_r_recall(real_gains, classifier_and_regressor_gains))
      avg_strict_r_recalls_certain.append(compute_r_recall(real_gains, classifier_certain_and_regressor_gains))
      avg_strict_r_recalls_classif_containment.append(compute_r_recall(real_gains, classifier_and_containment_gains))
      avg_strict_r_recalls_classif_certain_containment.append(compute_r_recall(real_gains, classifier_certain_and_containment_gains))
      avg_strict_r_recalls_classif_pearson.append(compute_r_recall(real_gains, classifier_and_max_pearson_diff_gains))
      avg_strict_r_recalls_classif_certain_pearson.append(compute_r_recall(real_gains, classifier_certain_and_max_pearson_diff_gains))
      avg_strict_r_recalls_containment.append(compute_r_recall(real_gains, containment_baseline_gains))
      avg_strict_r_recalls_baseline_regressor.append(compute_r_recall(real_gains, baseline_regressor_gains))

      numbers_of_retrieved_candidates.append(predicted_positive.shape[0])
  
  print('average number of candidates per query-target (predicted as positive)', np.mean(numbers_of_retrieved_candidates))
  print('Prec@5 - Classifier + Regressor:', np.mean(avg_precs))
  print('Prec@5 - Classifier (certain) + Regressor:', np.mean(avg_precs))
  print('Prec@5 - Classifier + Containment:', np.mean(avg_precs_classif_containment))
  print('Prec@5 - Classifier (certain) + Containment:', np.mean(avg_precs_classif_certain_containment))
  print('Prec@5 - Classifier + Max-Pearson-Diff:', np.mean(avg_precs_classif_pearson))
  print('Prec@5 - Classifier (certain) + Max-Pearson-Diff:', np.mean(avg_precs_classif_certain_pearson))  
  print('Prec@5 - Containment baseline:', np.mean(avg_precs_containment))
  print('Prec@5 - Regression baseline:', np.mean(avg_precs_baseline_regressor))
  
  print('Rec@5 - Classifier + Regressor:', np.mean(avg_recs))
  print('Rec@5 - Classifier (certain) + Regressor:', np.mean(avg_recs))
  print('Rec@5 - Classifier + Containment:', np.mean(avg_recs_classif_containment))
  print('Rec@5 - Classifier (certain) + Containment:', np.mean(avg_recs_classif_certain_containment))
  print('Rec@5 - Classifier + Max-Pearson-Diff:', np.mean(avg_recs_classif_pearson))
  print('Rec@5 - Classifier (certain) + Max-Pearson-Diff:', np.mean(avg_recs_classif_certain_pearson))
  print('Rec@5 - Containment baseline:', np.mean(avg_recs_containment))
  print('Rec@5 - Regression baseline:', np.mean(avg_recs_baseline_regressor))
  
  print('Strict R-Precision (k=5) - Classifier + Regressor:', np.mean(avg_strict_r_precisions))
  print('Strict R-Precision (k=5) - Classifier (certain) + Regressor:', np.mean(avg_strict_r_precisions))
  print('Strict R-Precision (k=5) - Classifier + Containment:', np.mean(avg_strict_r_precisions_classif_containment))
  print('Strict R-Precision (k=5) - Classifier (certain) + Containment:', np.mean(avg_strict_r_precisions_classif_certain_containment))
  print('Strict R-Precision (k=5) - Classifier + Max-Pearson-Diff:', np.mean(avg_strict_r_precisions_classif_pearson))
  print('Strict R-Precision (k=5) - Classifier (certain) + Max-Pearson-Diff:', np.mean(avg_strict_r_precisions_classif_certain_pearson))
  print('Strict R-Precision (k=5) - Containment baseline:', np.mean(avg_strict_r_precisions_containment))
  print('Strict R-Precision (k=5) - Regression baseline:', np.mean(avg_strict_r_precisions_baseline_regressor))

  print('Strict R-Recall (k=5) - Classifier + Regressor:', np.mean(avg_strict_r_recalls))
  print('Strict R-Recall (k=5) - Classifier (certain) + Regressor:', np.mean(avg_strict_r_recalls))
  print('Strict R-Recall (k=5) - Classifier + Containment:', np.mean(avg_strict_r_recalls_classif_containment))
  print('Strict R-Recall (k=5) - Classifier (certain) + Containment:', np.mean(avg_strict_r_recalls_classif_certain_containment))
  print('Strict R-Recall (k=5) - Classifier + Max-Pearson-Diff:', np.mean(avg_strict_r_recalls_classif_pearson))
  print('Strict R-Recall (k=5) - Classifier (certain) + Max-Pearson-Diff:', np.mean(avg_strict_r_recalls_classif_certain_pearson))
  print('Strict R-Recall (k=5) - Containment baseline:', np.mean(avg_strict_r_recalls_containment))
  print('Strict R-Recall (k=5) - Regression baseline:', np.mean(avg_strict_r_recalls_baseline_regressor))

  
if __name__ == '__main__':
  training_filename = sys.argv[1]
  test_filename = sys.argv[2]
  features = eval(open(sys.argv[3]).readline())
  baseline_regressor = pickle.load(open(sys.argv[4], 'rb'))
  
  training_data = pd.read_csv(training_filename)
  test_data = pd.read_csv(test_filename)
  test_with_predictions = generate_predictions(training_data, test_data, features)
  #analyze_predictions(test_with_predictions)
  regressor = build_regressor_for_ranking_positive_class(training_data, features)
  rank_candidates_classified_as_positive(test_with_predictions, regressor, features, baseline_regressor)
  
  
