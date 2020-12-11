'''
Given a list of poverty dataset ids, a list with ids to be used in an experiment, 
and the name of a folder for extracted files, this script (1) identifies where they are 
and (2) extracts them from binary files ans saves them in another folder. 
'''

import os
import codecs
import sys

def extract_and_save_dataset(dataset_id, reference, data_folder): 
    tokens = reference.split('-') 
    folder = tokens[0] 
    filename = tokens[1] + '-' + tokens[2].split(':')[0] 
    with codecs.open(os.path.join(folder, filename), encoding='utf-8', errors='ignore') as f: 
        line = f.readline() 
        while dataset_id not in line: 
            line = f.readline() 
        if ',' not in line: 
            line = f.readline() 
        new_file = codecs.open(os.path.join(data_folder, dataset_id), 'w', encoding='utf-8', errors='ignore') 
        new_file.write(line.split('\x00')[-1]) 
        line = f.readline() 
        while '\x00' not in line and 'q' not in line: 
            new_file.write(line) 
            line = f.readline() 
        new_file.close() 

def detect_datasets(dataset_ids, dataset_ids_to_find, new_folder): 
    not_identified = [] 
    for id_ in dataset_ids_to_find: 
        found = False 
        for line in dataset_ids: 
            if id_ in line and 'part-' in line: 
                extract_and_save_dataset(id_, line, new_folder) 
                found = True
                break
        if not found: 
            not_identified.append(id_) 
    return not_identified 


import os

if __name__ == '__main__':

    all_datasets = codecs.open(sys.argv[1], 'r', encoding='utf-8', errors='ignore').readlines()
    datasets_to_find = [elem.strip() for elem in codecs.open(sys.argv[2], 'r', encoding='utf-8', errors='ignore').readlines()]
    folder_for_datasets = sys.argv[3]

    not_identified = detect_datasets(all_datasets, datasets_to_find, folder_for_datasets)

    f = open('not_identified_' + os.path.basename(folder_for_datasets) + '.txt', 'w')
    for dataset in not_identified:
        f.write(dataset + '\n')
    f.close()
