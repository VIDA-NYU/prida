""" Given (1) two files (one with training and one with test data; both with headers), 
          (2) a threshold alpha above which a gain in R2 squared should correspond to class GOOD GAIN, and 
          (3) a file with the features that should be used for learning, 

          this script explores different ways of combining classification results with other sources of info 
          in order to recommend useful datasets for augmentation. 
"""

import sys
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from sklearn.metrics import f1_score, classification_report
from scipy.stats import pearsonr
import numpy as np

TARGET_COLUMN = 'gain_in_r2_score'
POSITIVE_CLASS = 'good_gain'
NEGATIVE_CLASS = 'loss'

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


def determine_classes_based_on_gain_in_r2_score(dataset, alpha, downsample=True):
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
    candidates_per_query_target = {row['query'] + row['target']: {} for index, row in dataset_with_predictions.iterrows()}
    for index, row in dataset_with_predictions.iterrows():
        key = row['query'] + row['target']
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
    num_candidates = []
    for key in candidates_per_query_target.keys():
        candidates = candidates_per_query_target[key].keys()
        positive_right = 0
        for candidate in candidates: #TODO what if there are no true positives?
            if candidates_per_query_target[key][candidate][TARGET_COLUMN] > alpha and candidates_per_query_target[key][candidate]['class'] == POSITIVE_CLASS:
                positive_right += 1
        num_candidates.append(len(candidates))
        precs.append(positive_right/len(candidates))
    print('correlation between precision and number of candidates', pearsonr(num_candidates, precs))
    return precs

def compute_recall_for_top_k_candidates(candidates_per_query_target, alpha, k):
    """This function computes how many of the top k candidates we efficiently retrieve
    """
    top_recall = []
    num_cands = []
    for key in candidates_per_query_target.keys():
        candidates = candidates_per_query_target[key].keys()
        num_cands.append(len(candidates))
        gains = []
        for candidate in candidates:
            gains.append((candidates_per_query_target[key][candidate][TARGET_COLUMN], candidates_per_query_target[key][candidate]['class']))
        relevant_gains = [i for i in sorted(gains)[-k:] if i[0] > alpha]
        positive_right = 0
        for (gain, class_) in relevant_gains:
            if class_ == POSITIVE_CLASS:
                positive_right += 1
        if len(relevant_gains):
            top_recall.append(positive_right/min(k, len(relevant_gains)))
    return top_recall
    
def analyze_predictions(test_with_preds, alpha):
    """This function separates all candidates for each 
    query-target pair and then analyzes how well the classification worked in 
    each case
    """
    candidates_per_query_target = parse_rows(test_with_preds)
    print('correlation between the probability of being in the positive class and the actual gains', compute_correlation_prob_class_target(candidates_per_query_target))
    #print('average precision for positive class per query-target', np.mean(compute_precision_per_query_target(candidates_per_query_target, alpha)))
    print('What is the average recall for the top-5 candidates?', np.mean(compute_recall_for_top_k_candidates(candidates_per_query_target, alpha, 5)))
    print('What is the average recall for the top-1 candidates?', np.mean(compute_recall_for_top_k_candidates(candidates_per_query_target, alpha, 1)))
    print('What is the average recall for the top-3 candidates?', np.mean(compute_recall_for_top_k_candidates(candidates_per_query_target, alpha, 3)))
    #Analysis 3: are candidates with really high gain often predicted as such? i.e., are we missing them (low recall right there?)
    
if __name__ == '__main__':
    training_filename = sys.argv[1]
    test_filename = sys.argv[2]
    alpha = float(sys.argv[3])
    features = eval(open(sys.argv[4]).readline())

    test_with_predictions = generate_predictions(pd.read_csv(training_filename), pd.read_csv(test_filename), alpha, features)
    analyze_predictions(test_with_predictions, alpha)
