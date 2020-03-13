import sys
import os
import shutil
import json
import uuid


def create_new_dir(dir_):
    new_dir = os.path.join(
        os.path.dirname(dir_),
        os.path.basename(dir_) + '-id'
    )
    if os.path.exists(new_dir):
        shutil.rmtree(new_dir)
    os.mkdir(new_dir)
    return new_dir


companion_datasets_dir = sys.argv[1]
records_directories = list()
for i in range(2, len(sys.argv)):
    records_directories.append(sys.argv[i])

new_companion_datasets_dir = create_new_dir(companion_datasets_dir)

name_to_id = dict()
for dir_ in records_directories:
    new_dir = create_new_dir(dir_)
    records_file = open(os.path.join(dir_, 'datamart-records'))
    new_records_file = open(os.path.join(new_dir, 'datamart-records'), 'w')
    line = records_file.readline()
    while line != '':
        data = json.loads(line)
        candidate_data = data['candidate_dataset']
        candidate_path_elems = candidate_data.split(os.path.sep)
        dataset_name = os.path.join(*candidate_path_elems[-2:])
        if not os.path.exists(dataset_name):
            line = records_file.readline()
            continue
        if dataset_name not in name_to_id:
            name_to_id[dataset_name] = str(uuid.uuid4())
        candidate_path_elems[-2] = new_companion_datasets_dir
        candidate_path_elems[-1] = name_to_id[dataset_name]
        data['candidate_dataset'] = os.path.join(*candidate_path_elems)
        new_records_file.write(json.dumps(data) + "\n")
        line = records_file.readline()
    records_file.close()
    new_records_file.close()

for k in name_to_id:
    shutil.copyfile(k, os.path.join(new_companion_datasets_dir, name_to_id[k]))

