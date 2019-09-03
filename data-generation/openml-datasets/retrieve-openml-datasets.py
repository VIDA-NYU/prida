import json
import openml
import os
import pandas as pd
import shutil
import sys


def save_dataset(directory, dataset, task):
    """Saves an OpenML dataset in D3M (minimal) format
    so that we can perform data generation.
    """

    X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format='dataframe'
    )
    numeric_attributes = dataset.get_features_by_type('numeric')
    target_id = attribute_names.index(task['target_feature'])

    if target_id not in numeric_attributes:
        return

    # creating directory
    dir_name = os.path.join(directory, dataset.name)
    dataset_dir_name = os.path.join(dir_name, dataset.name + '_dataset')
    tables_dir_name = os.path.join(dataset_dir_name, 'tables')
    problem_dir_name = os.path.join(dir_name, dataset.name + '_problem')
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)
    os.mkdir(dir_name)
    os.mkdir(dataset_dir_name)
    os.mkdir(tables_dir_name)
    os.mkdir(problem_dir_name)

    # saving dataset
    X.to_csv(
        open(os.path.join(tables_dir_name, 'learningData.csv'), 'w'),
        index=False
    )

    # datasetDoc.json
    dataset_doc = dict(dataResources=[])
    learning_data_resource = {
        'resID': 'learningData',
        'resPath': 'tables/learningData.csv',
        'columns': []
    }
    for i in range(len(attribute_names)):
        learning_data_resource['columns'].append(
            {
                'colIndex': i,
                'colName': attribute_names[i],
                'colType': 'real' if i in numeric_attributes else 'categorical'
            }
        )
    dataset_doc['dataResources'].append(learning_data_resource)
    with open(os.path.join(dataset_dir_name, 'datasetDoc.json'), 'w') as f:
        json.dump(dataset_doc, f, indent=4)

    # problemDoc.json
    problem_doc = dict(
        about=dict(taskType='regression'),
        inputs=dict(data=[])
    )
    problem_doc['inputs']['data'].append(dict(targets=[]))
    problem_doc['inputs']['data'][0]['targets'].append(
        dict(
            colName=task['target_feature'],
            colIndex=target_id
        )
    )
    with open(os.path.join(problem_dir_name, 'problemDoc.json'), 'w') as f:
        json.dump(problem_doc, f, indent=4)


if __name__ == '__main__':
    output_dir = sys.argv[1]
    regression_tasks = openml.tasks.list_tasks(task_type_id=2)
    seen_datasets = set()
    for task_id in regression_tasks:
        task = regression_tasks[task_id]
        try:
            dataset = openml.datasets.get_dataset(task['source_data'])
        except openml.exceptions.OpenMLServerException as e:
            print("Dataset skipped due to OpenML exception: %d" % task['source_data'])
            continue
        except Exception as e:
            print("Dataset skipped due to an exception: %d" % task['source_data'])
            print("  Exception on dataset: %s" % str(e))
            continue
        if dataset.name in seen_datasets:
            continue
        seen_datasets.add(dataset.name)
        print("Saving dataset %s" % dataset.name)
        try:
            save_dataset(output_dir, dataset, task)
        except Exception as e:
            print("  Exception on dataset: %s" % str(e))
       