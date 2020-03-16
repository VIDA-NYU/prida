""" This script evaluates the classification performance of different dataset draws in the formats:
      (1) draw_*_one_candidate_per_query_from_training-simplified-data-generation-many-candidates-per-query_with_median_and_mean_based_classes.csv
      (2) draw_*_two_candidated_per_query_from_training-simplified-data-generation-many-candidates-per-query_with_median_and_mean_based_classes.csv
    over college and taxi use cases
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

FEATURES = ['query_num_of_columns', 'query_num_of_rows', 'query_row_column_ratio', 'query_max_skewness', 'query_max_kurtosis', 'query_max_unique', 'candidate_num_rows', 'candidate_row_column_ratio', 'candidate_max_skewness', 'candidate_max_kurtosis', 'candidate_max_unique', 'query_target_max_pearson', 'query_target_max_spearman', 'query_target_max_covariance', 'query_target_max_mutual_info', 'candidate_target_max_pearson', 'candidate_target_max_spearman', 'candidate_target_max_covariance', 'candidate_target_max_mutual_info', 'max_pearson_difference', 'containment_fraction']
NUM_DATASETS_ONE_CANDIDATE = 5
NUM_DATASETS_TWO_CANDIDATES = 5


if __name__=='__main__':
  college = pd.read_csv('college-debt-records-features-single-column-w-class_with_median_and_mean_based_classes.csv')
  college['class'] =  ['good_gain' if row['gain_in_r2_score'] > 0 else 'loss' for index, row in college.iterrows()]
  
  taxi = pd.read_csv('taxi-vehicle-collision-records-features-single-column-w-class_with_median_and_mean_based_classes.csv')
  taxi['class'] =  ['good_gain' if row['gain_in_r2_score'] > 0 else 'loss' for index, row in taxi.iterrows()]

  for i in range(NUM_DATASETS_ONE_CANDIDATE):
    training = pd.read_csv('draw_'+str(i)+'_one_candidate_per_query_from_training-simplified-data-generation-many-candidates-per-query_with_median_and_mean_based_classes.csv')
    training['class'] = ['good_gain' if row['gain_in_r2_score'] > 0 else 'loss' for index, row in training.iterrows()]
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(training[FEATURES], training['class'])
    taxi_preds = rf.predict(taxi[FEATURES])
    taxi_preds_proba = rf.predict_proba(taxi[FEATURES])
    print('taxi report for draw_' + str(i) + '_one_candidate_per_query')
    print(classification_report(taxi['class'], taxi_preds))
    print('taxi report for draw_' + str(i) + '_one_candidate_per_query IGNORING PREDS PROBS IN [0.4, 0.6]')
    tests = []; preds = []
    for pred, prob, test in zip(taxi_preds, taxi_preds_proba, taxi['class']):
      if prob[0] < 0.4 or prob[0] > 0.6:
        tests.append(test)
        preds.append(pred)       
    print(classification_report(tests, preds))

    college_preds = rf.predict(college[FEATURES])
    college_preds_proba = rf.predict_proba(college[FEATURES])
    print('college report for draw_' + str(i) + '_one_candidate_per_query')
    print(classification_report(college['class'], college_preds))
    print('college report for draw_' + str(i) + '_one_candidate_per_query IGNORING PREDS PROBS IN [0.4, 0.6]')
    tests = []; preds = []
    for pred, prob, test in zip(college_preds, college_preds_proba, college['class']):
      if prob[0] < 0.4 or prob[0] > 0.6:
        tests.append(test)
        preds.append(pred)       
    print(classification_report(tests, preds))

  for i in range(NUM_DATASETS_TWO_CANDIDATES):
    training = pd.read_csv('draw_'+str(i)+'_two_candidates_per_query_from_training-simplified-data-generation-many-candidates-per-query_with_median_and_mean_based_classes.csv')
    training['class'] = ['good_gain' if row['gain_in_r2_score'] > 0 else 'loss' for index, row in training.iterrows()]
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(training[FEATURES], training['class'])
    taxi_preds = rf.predict(taxi[FEATURES])
    taxi_preds_proba = rf.predict_proba(taxi[FEATURES])
    print('taxi report for draw_' + str(i) + '_two_candidates_per_query')
    print(classification_report(taxi['class'], taxi_preds))
    print('taxi report for draw_' + str(i) + '_two_candidates_per_query IGNORING PREDS PROBS IN [0.4, 0.6]')
    tests = []; preds = []
    for pred, prob, test in zip(taxi_preds, taxi_preds_proba, taxi['class']):
      if prob[0] < 0.4 or prob[0] > 0.6:
        tests.append(test)
        preds.append(pred)       
    print(classification_report(tests, preds))

    college_preds = rf.predict(college[FEATURES])
    college_preds_proba = rf.predict_proba(college[FEATURES])
    print('college report for draw_' + str(i) + '_two_candidates_per_query')
    print(classification_report(college['class'], college_preds))
    print('college report for draw_' + str(i) + '_two_candidates_per_query IGNORING PREDS PROBS IN [0.4, 0.6]')
    tests = []; preds = []
    for pred, prob, test in zip(college_preds, college_preds_proba, college['class']):
      if prob[0] < 0.4 or prob[0] > 0.6:
        tests.append(test)
        preds.append(pred)       
    print(classification_report(tests, preds))

    
