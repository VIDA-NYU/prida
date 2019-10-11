import os
import subprocess
import sys


if __name__ == '__main__':
    original_data_dir = sys.argv[1]
    hdfs_data_dir = sys.argv[2]
    if hdfs_data_dir.endswith('/'):
        hdfs_data_dir = hdfs_data_dir[:-1]

    if hdfs_data_dir.strip() == '':
        sys.exit(0)

    # Making sure HDFS dir is empty
    subprocess.call('hdfs dfs -rm -r %s/*' % hdfs_data_dir, shell=True)

    # Copying files
    datasets_ = os.listdir(original_data_dir)
    datasets = list()
    for dataset in datasets_:
        data_path = os.path.join(original_data_dir, dataset, dataset + '_dataset', 'tables', 'learningData.csv')
        data_size = os.stat(data_path).st_size/float(1073741824)
        if data_size <= 2:
            datasets.append(dataset)
    
    count = 1
    for dataset in datasets:

        print("Uploading %s ... (%.4f%%)" % (dataset, (float(count)*100)/len(datasets)))

        dataset_doc = os.path.join(original_data_dir, dataset, dataset + '_dataset', 'datasetDoc.json')
        dataset_path = os.path.join(original_data_dir, dataset, dataset + '_dataset', 'tables', 'learningData.csv')
        problem_doc = os.path.join(original_data_dir, dataset, dataset + '_problem', 'problemDoc.json')

        subprocess.call('hdfs dfs -mkdir \'%s/%s\'' % (hdfs_data_dir, dataset), shell=True)
        subprocess.call('hdfs dfs -put \'%s\' \'%s/%s/\'' % (dataset_doc, hdfs_data_dir, dataset), shell=True)
        subprocess.call('hdfs dfs -put \'%s\' \'%s/%s/\'' % (problem_doc, hdfs_data_dir, dataset), shell=True)
        subprocess.call('hdfs dfs -put \'%s\' \'%s/%s/\'' % (dataset_path, hdfs_data_dir, dataset), shell=True)

        count += 1

