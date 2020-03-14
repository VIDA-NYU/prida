""" Given 
      (1) a large dataset from which we can sample new, smaller 'datasets'
      (2) the number of distinct query datasets that should be used when composing the new dataset
    this script explores different ways of drawing samples that correspond to these 'datasets'. The idea is to draw several datasets following the same mechanism 
    to verify whether we get similar results over our test instances (i.e. whether this dataset generation mechanism is robust). Here, we draw samples from a larger 
    dataset,  where each query Qi is randomly paired with multiple candidate datasets from different 'source datasets' (openml 'original' datasets, for example). 
"""


import pandas as pd
import sys
import numpy as np

NUMBER_OF_VERSIONS_WITH_ONE_CANDIDATE_PER_QUERY = 5
NUMBER_OF_VERSIONS_WITH_TWO_CANDIDATES_PER_QUERY = 5

def create_version_of_dataset_2(larger_dataset, n_queries, one_candidate_per_query=True):
  """This function draws candidates from larger_dataset for n_queries of its queries. 
  
  If one_candidate_per_query == True, it only draws one candidate, with either 
  gain_marker == 'positive' or gain_marker == 'negative', per query. Otherwise, it 
  draws two candidates (one with gain_marker == 'positive' and one with gain_marker == 'negative')
  """
  
  queries = np.random.choice(list(set(larger_dataset['query'])), n_queries)
  subdatasets = []
  i = 0
  for q in queries:
    if i % 1000 == 0:
      print(i)
    i += 1  
    subtable = larger_dataset.loc[larger_dataset['query'] == q]
    if one_candidate_per_query:
      sample = subtable.sample(1)
    else:
      positives = subtable.loc[subtable['gain_marker'] == 'positive']
      sample_positive = positives.sample(1)
      negatives = subtable.loc[subtable['gain_marker'] == 'negative']
      sample_negative = negatives.sample(1)
      sample = pd.concat([sample_positive, sample_negative])
    subdatasets.append(sample)
  return pd.concat(subdatasets)

if __name__ == '__main__':

  dataset_name = sys.argv[1]
  number_of_queries = int(sys.argv[2])
  dataset = pd.read_csv(dataset_name)

  for i in range(NUMBER_OF_VERSIONS_WITH_ONE_CANDIDATE_PER_QUERY):
    drawn_dataset = create_version_of_dataset_2(dataset, number_of_queries)
    f = open('draw_' + str(i) + '_one_candidate_per_query_from_' + dataset_name, 'w')
    f.write(drawn_dataset.to_csv(index=False))
    f.close()
  for i in range(NUMBER_OF_VERSIONS_WITH_TWO_CANDIDATES_PER_QUERY):
    drawn_dataset = create_version_of_dataset_2(dataset, number_of_queries, one_candidate_per_query=False)
    f = open('draw_' + str(i) + '_two_candidates_per_query_from_' + dataset_name, 'w')
    f.write(drawn_dataset.to_csv(index=False))
    f.close()
  
