'''
This script downloads datasets from Auctus that are joinable with a given table. To this end, one must specify:

   argv[1] => filename for the base table
   argv[2] => an attribute in the base table that should be used as the key
'''

import sys
import io
import json
import pandas
from pprint import pprint
import requests
import zipfile
   
def download_joinable_datasets(base_table, key):
     URL = 'https://auctus.vida-nyu.org/api/v1'
     data = base_table
     url = URL + '/search'
     query = {"augmentation_type": "join"}
     with open(data, 'rb') as data_p:
       response = requests.post(
         url,
         files={
             'data': data_p,
             #'query': ('query.json', json.dumps(query), 'application/json'),
         }
       )
     query_results = response.json()['results']
     print(len(query_results))
     url = URL + '/download'
     for result in query_results:
         id_ = result['id']
         print(id_, result['augmentation']['right_columns_names'][0][0])
         aug_type = result['augmentation']['type']
         left_column = result['augmentation']['left_columns_names'][0][0]
         if aug_type == 'join' and left_column == key:
             right_column = result['augmentation']['right_columns_names'][0][0]
             response = requests.get(url + '/%s' % id_, params={'format': 'd3m'}, timeout=100)
             if response.status_code == 400:
               try:
                 print('Error: %s' % response.json()['error'])
               except:
                 print('exception')
                 pass
             #response.raise_for_status()
             try:
               zip_ = zipfile.ZipFile(BytesIO(response.content), 'r')
               learning_data = pd.read_csv(zip_.open('tables/learningData.csv'))
               f = open(id_, 'w')
               f.write(learning_data.rename(columns={right_column: left_column}).to_csv(index=False))
               f.close()
             except:
               continue

if __name__=='__main__':
    base_table = sys.argv[1]
    key = sys.argv[2]
    download_joinable_datasets(base_table, key)
