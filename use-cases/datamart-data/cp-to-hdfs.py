import os
import subprocess
import sys


if __name__ == '__main__':
    hdfs_data_dir = sys.argv[1]
    if hdfs_data_dir.endswith('/'):
        hdfs_data_dir = hdfs_data_dir[:-1]

    if hdfs_data_dir.strip() == '':
        sys.exit(0)

    # Making sure HDFS dir is empty
    subprocess.call('hdfs dfs -rm -r %s/*' % hdfs_data_dir, shell=True)

    # Getting all the paths
    datasets_ = os.listdir('companion-datasets-id')
    datasets = list()
    for dataset in datasets_:
        path = os.path.join('companion-datasets-id', dataset)
        if not os.path.isdir(path):
            datasets.append(path)
    datasets_ = os.listdir('companion-datasets')
    join_dirs = dict()
    for dataset in datasets_:
        path = os.path.join('companion-datasets', dataset)
        if os.path.isdir(path):
            join_dirs[dataset] = list()
    for dataset in join_dirs:
        path = os.path.join('companion-datasets', dataset)
        for join_file in os.listdir(path):
            join_dirs[dataset].append(os.path.join(path, join_file))
    query_datasets = list()
    query_datasets.append("../data/college-debt/college-debt-v2.csv")
    query_datasets.append("../data/taxi-vehicle-collision/taxi-vehicle-collision-v2.csv")
    query_datasets.append("../data/poverty-estimation/poverty-estimation-v2.csv")
   
    # Uploading query datasets
    query_path = os.path.join(hdfs_data_dir, 'query-datasets')
    subprocess.call('hdfs dfs -mkdir %s' % query_path, shell=True)
    count = 1
    print("Uploading query datasets ...\n")
    for dataset in query_datasets:
        print("Uploading %s ... (%.4f%%)" % (os.path.basename(dataset), (float(count)*100)/len(query_datasets)))
        subprocess.call('hdfs dfs -put \'%s\' \'%s/\'' % (dataset, query_path), shell=True)
        count += 1
 
    # Uploading companion datasets
    companion_path = os.path.join(hdfs_data_dir, 'companion-datasets')
    subprocess.call('hdfs dfs -mkdir %s' % companion_path, shell=True)
    count = 1
    print("Uploading companion datasets ...\n")
    for dataset in datasets:
        print("Uploading %s ... (%.4f%%)" % (os.path.basename(dataset), (float(count)*100)/len(datasets)))
        subprocess.call('hdfs dfs -put \'%s\' \'%s/\'' % (dataset, companion_path), shell=True)
        count += 1

    # Uploading join datasets
    join_path = os.path.join(hdfs_data_dir, 'join-datasets')
    subprocess.call('hdfs dfs -mkdir %s' % join_path, shell=True)
    print("\nUploading join datasets ...\n")
    for dir_ in join_dirs:
        count = 1
        join_subpath = os.path.join(join_path, dir_)
        subprocess.call('hdfs dfs -mkdir %s' % join_subpath, shell=True)
        print("Uploading join datasets (%s) ..." % dir_)
        for dataset in join_dirs[dir_]:
            print("Uploading %s ... (%.4f%%)" % (os.path.basename(dataset), (float(count)*100)/len(join_dirs[dir_])))
            subprocess.call('hdfs dfs -put \'%s\' \'%s/\'' % (dataset, join_subpath), shell=True)
            count += 1

    # Uploading training records
    path = os.path.join(hdfs_data_dir, 'taxi-vehicle-collisions-training-records')
    subprocess.call('hdfs dfs -put taxi-vehicle-collision-datamart-records-id/datamart-records  \'%s/\'' % path, shell=True)
    path = os.path.join(hdfs_data_dir, 'college-debt-training-records')
    subprocess.call('hdfs dfs -put college-debt-datamart-records-id/datamart-records  \'%s/\'' % path, shell=True)
    path = os.path.join(hdfs_data_dir, 'poverty-estimation-training-records')
    subprocess.call('hdfs dfs -put poverty-estimation-datamart-records-id/datamart-records  \'%s/\'' % path, shell=True)

