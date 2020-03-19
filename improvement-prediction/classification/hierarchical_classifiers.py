""" Given 
      (1) a dataset for training,
      (2) a dataset for the taxi use case, and
      (3) a dataset for the college use case
    this script explores different ways of using classifiers in a hierarchical fashion. 
"""


import pandas as pd
import sys
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier


CANDIDATE_FEATURES = ['candidate_num_of_columns', 'candidate_num_rows', 'candidate_row_column_ratio', 'candidate_max_mean', 'candidate_max_outlier_percentage', 
                      'candidate_max_skewness', 'candidate_max_kurtosis', 'candidate_max_unique']
CANDIDATE_TARGET_FEATURES = ['candidate_target_max_pearson', 'candidate_target_max_spearman', 'candidate_target_max_covariance', 'candidate_target_max_mutual_info']
MISC_FEATURES = ['max_pearson_difference', 'containment_fraction']
QUERY_FEATURES = ['query_num_of_columns', 'query_num_of_rows', 'query_row_column_ratio', 'query_max_mean', 'query_max_outlier_percentage', 'query_max_skewness',
                  'query_max_kurtosis', 'query_max_unique']
QUERY_TARGET_FEATURES = ['query_target_max_pearson', 'query_target_max_spearman', 'query_target_max_covariance', 'query_target_max_mutual_info']

CLASSIFIER_NAME_ID = 0
CLASSIFIER_MODEL_ID = 1
CLASSIFIER_FEATURES_ID = 2

def create_classifier(training_data, features, target_column):
  """ This function creates a random forest classifier based on 
  given training data, features, and a target_column indicating classes
  """ 
  rf = RandomForestClassifier(n_estimators=100, random_state=42)
  rf.fit(training_data[features], training_data[target_column])
  return rf

def combine_classifiers(first_classifier, second_classifier, test_data):
  """ This function uses first_classifier to generate predictions for test_data and, 
  for every instance that gets classified as good_gain (positive), it generates a 
  second prediction with second_classifier.

  If an instance is classified as positive by both classifiers, we consider the prediction 
  as 'good_gain'; otherwise, as loss.
  """
  
  first_preds = first_classifier[CLASSIFIER_MODEL_ID].predict(test_data[first_classifier[CLASSIFIER_FEATURES_ID]])
taxi_preds_0[0:10]
instaces_for_classifier_1 = []
for pred, instance in zip(taxi_preds_0, taxi):
    if pred == 'good_gain':
        instances_for_classifier_1.append(instance)
        
for pred, instance in zip(taxi_preds_0, taxi):
    if pred == 'good_gain':
        instaces_for_classifier_1.append(instance)
        
taxi_preds_1 = query_everything_classif.predict(pd.concat([QUERY_FEATURES + QUERY_TARGET_FEATURES]))
instaces_for_classifier_1 = pd.concat(instaces_for_classifier_1)
instaces_for_classifier_1
instances_for_classifier_1 = []
taxi_preds_0 = candidate_everything_and_misc_classifier.predict(taxi[CANDIDATE_FEATURES + CANDIDATE_TARGET_FEATURES + MISC_FEATURES])
len(taxi_preds_0)
for pred, instance in zip(taxi_preds_0, taxi):
    if pred == 'good_gain':
        instances_for_classifier_1.append(instance)
        
instances_for_classifier_1
for pred, instance in zip(taxi_preds_0, taxi):
    print(pred, instance)
    if pred == 'good_gain':
        instances_for_classifier_1.append(instance)
        
for index, row in taxi.iterrows():
    if taxi_preds_0[index] == 'good_gain':
        instaces_for_classifier_1.append(row)
        
for index, row in taxi.iterrows():
    if taxi_preds_0[index] == 'good_gain':
        instances_for_classifier_1.append(row)
        
instances_for_classifier_1 = pd.concat(instances_for_classifier_1)
instances_for_classifier_1
instances_for_classifier_1[0]
instances_for_classifier_1 = []
for index, row in taxi.iterrows():
    if taxi_preds_0[index] == 'good_gain':
        instances_for_classifier_1.append(row)
        
instances_for_classifier_1[0]
instances_for_classifier_1 = pd.concat(instances_for_classifier_1)
taxi_preds_1 = query_everything_classif.predict(instances_for_classifier_1[QUERY_FEATURES + QUERY_TARGET_FEATURES])
instances_for_classifier_1.shape
instances_for_classifier_1
instances_for_classifier_1 = []
for index, row in taxi.iterrows():
    if taxi_preds_0[index] == 'good_gain':
        instances_for_classifier_1.append(row)
        
len(instances_for_classifier_1)
pd.DataFrame(instances_for_classifier_1).shape
instances_for_classifier_1 = pd.DataFrame(instances_for_classifier_1)
instances_for_classifier_1.shape
taxi_preds_1 = query_everything_classif.predict(instances_for_classifier_1[QUERY_FEATURES + QUERY_TARGET_FEATURES])
num_positive_preds = len([pred for pred in taxi_preds_1 if pred == 'good_gain'])
num_positive_preds
num_true_positive = []
instances_for_classifier_1
instances_for_classifier_1.index
for index, row in instances_for_classifier_1.iterrows():
    print(index,)
    
i = 0
instances_for_classifier_1[434]
instances_for_classifier_1.iloc[434]
instances_for_classifier_1.iloc(434)
instances_for_classifier_1.loc[434]
taxi.shape
final_predictions = []
for index in range(447):
    try:
        row = instances_for_classifier_1.loc[index]
        final_predictions.append(taxi_preds_1[i])
    except:
        final_predictions.append('loss')
    i += 1
            
len(final_predictions)
i
classification_report(taxi['class'], final_predictions)
print(classification_report(taxi['class'], final_predictions))
%save -r hierarchical_classifiers 1-80

  
if __name__=='__main__':
  training_data = pd.read_csv(sys.argv[1])
  training_data['class'] = ['good_gain' if row['gain_in_r2_score'] > 0 else 'loss' for index, row in training_data.iterrows()]
  taxi = pd.read_csv(sys.argv[2])
  taxi['class'] = ['good_gain' if row['gain_in_r2_score'] > 0 else 'loss' for index, row in taxi.iterrows()]
  college = pd.read_csv(sys.argv[3]) 
  college['class'] = ['good_gain' if row['gain_in_r2_score'] > 0 else 'loss' for index, row in college.iterrows()]

  candidate_classif = create_classifier(training_data, CANDIDATE_FEATURES, 'class')
  candidate_target_classif = create_classifier(training_data, CANDIDATE_TARGET_FEATURES, 'class')
  candidate_everything_classif = create_classifier(training_data, CANDIDATE_FEATURES + CANDIDATE_TARGET_FEATURES, 'class')
  target_classif = create_classifier(training_data, target_features, 'class')
  query_classif = create_classifier(training_data, QUERY_FEATURES, 'class')
  query_target_classif = create_classifier(training_data, QUERY_TARGET_FEATURES, 'class')
  query_everything_classif = create_classifier(training_data, QUERY_FEATURES + QUERY_TARGET_FEATURES, 'class')
  misc_classifier = create_classifier(training_data, MISC_FEATURES, 'class')
  candidate_everything_and_misc_classifier = create_classifier(training_data, CANDIDATE_FEATURES + CANDIDATE_TARGET_FEATURES + MISC_FEATURES, 'class')

  classifiers = [('candidate_classif', candidate_classif, CANDIDATE_FEATURES), ('candidate_target_classif', candidate_target_classif, CANDIDATE_TARGET_FEATURES), 
                 ('candidate_everything_classif', candidate_everything_classif, CANDIDATE_FEATURES + CANDIDATE_TARGET_FEATURES), 
                 ('query_classif', query_classif, QUERY_FEATURES), ('query_target_classif', query_target_classif, QUERY_TARGET_FEATURES), 
                 ('query_everything_classif', query_everything_classif, QUERY_FEATURES + QUERY_TARGET_FEATURES), ('misc_classifier', misc_classifier, MISC_FEATURES), 
                 ('candidate_everything_and_misc_classifier', candidate_everything_and_misc_classifier, CANDIDATE_FEATURES + CANDIDATE_TARGET_FEATURES + MISC_FEATURES)]

  report = combine_classifiers(classifiers[7], classifiers[5], taxi)
  print(report)

