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
        
def analyze_predictions(test_with_preds):
    """This function separates all candidates for each 
    query-target pair and then analyzes how well the classification worked in 
    each case
    """
    candidates_per_query_target = parse_rows(test_with_preds)
    print(candidates_per_query_target['dataset-ranking/openml-datasets-single-column-results/files/80f5e205-f72c-4dde-a9dc-7fd17f376af2/query_SensIT-Vehicle-Combined_0.csvclass'])
    #Analysis 1: do prob_positive_class correlate with the target column?
    #Analysis 2: are we not classifying candidates with bad gain as candidates with good gain?
    #Analysis 3: are candidates with really high gain often predicted as such? i.e., are we missing them (low recall right there?)
    
if __name__ == '__main__':
    training_filename = sys.argv[1]
    test_filename = sys.argv[2]
    alpha = float(sys.argv[3])
    features = eval(open(sys.argv[4]).readline())

    test_with_predictions = generate_predictions(pd.read_csv(training_filename), pd.read_csv(test_filename), alpha, features)
    analyze_predictions(test_with_predictions)
