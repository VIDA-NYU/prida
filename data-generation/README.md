# Generation of Training Data

## Requirements

* [Python 3](https://www.python.org/)
* [HdfsCLI](https://hdfscli.readthedocs.io/en/latest/)
* [NumPy](https://numpy.org/)
* [pandas](https://pandas.pydata.org/)
* [scikit-learn](https://scikit-learn.org/stable/)
* [SciPy](https://www.scipy.org/)
* [XGBoost](https://xgboost.readthedocs.io/en/latest/python/python_intro.html)
* [Apache Spark 2.3.0](https://spark.apache.org/)
* [Matplotlib](https://matplotlib.org/) (for plots only)

## Generating Data

The data generation process is done using PySpark. Copy the file [`params.json`](params.json), name it as `.params.json`, and configure it appropriately. The structure of this file is the following:

```
{
    "original_datasets_directory": directory of original datasets
    "new_datasets_directory": output directory where the query and the candidate datasets will be saved
    "cluster": boolean that indicates whether the data generation process will be run in cluster or not
    "hdfs_address": the address (host and port) for the distributed file system; only used if the data generation will be run in a cluster
    "hdfs_user": the username for the distributed file system; only used if the data generation will be run in a cluster
    "ignore_first_attribute": boolean that indicates whether the first attribute of every dataset should be ignored (e.g.: the d3mIndex attribute)
    "skip_dataset_creation": boolean that indicates whether the generation of query and candidate datasets should be skipped or not
    "skip_training_data": boolean that indicates whether the generation of the training data should be skipped or not
    "candidate_single_column": boolean that indicates whether the candidate datasets should have a single column (in addition to the key column) or not
    "regression_algorithm": the regression algorithm to be used; the available options are "random forest", "linear", "sgd", and "xgboost"
    "inner_join": boolean that indicates whether the join applied between query and candidate datasets is of type inner or not; if false, a left join is applied and a series of univariate value imputation strategies are applied to take care of any missing values, with the one that generates the model with the smallest mean absolute error being chosen at last
    "min_number_records": the minimum number of records that a query or candidate dataset should have
    "max_number_columns": the maximum number of columns that a query/candidate dataset pair should have
    "max_times_break_data_vertical": the maximum number of times that a dataset will be broken (vertically) into multiple data
    "max_times_records_removed": the maximum number of times that records will be removed from a dataset to derive new data
    "max_percentage_noise": the maximum percentage of records from the candidate dataset that will be replaced by gaussian noise
}
```

### Client Mode

To run the data generation process locally, run the following:

    $ spark-submit \
    --deploy-mode client \
    --files .params.json \
    generate-training-data-from-datasets.py

You may need to set some parameters for `spark-submit` depending on your environment. For an example, you can inspect the script [`run-spark-client`](run-spark-client).

### Cluster Mode (Apache YARN)

The easiest way to run the data generation in a cluster is by using [Anaconda](https://www.anaconda.com/) to package the python dependencies. First, install Anaconda and initialize it by using the `conda init` command. Then, run the following:

    $ conda create -y -n data-generation -c conda-forge python=3.6.9 numpy pandas scikit-learn scipy python-hdfs xgboost
    $ cd <env_dir>
    $ zip -r <data-generation-dir>/data-generation-environment.zip data-generation/

where `<env_dir>` is the location for Anaconda environments, and `<data-generation-dir>` is the location for this directory. The `data-generation-environment.zip` file will contain all the packages necessary to run the data generation script, making sure that all of the cluster nodes have access to the dependencies. To submit the data generation job, run the following:

    $ cd <data-generation-dir>
    $ spark-submit \
    --deploy-mode cluster \
    --master yarn \
    --files .params.json \
    --conf spark.yarn.appMasterEnv.PYSPARK_PYTHON=./env/data-generation/bin/python \
    --archives data-generation-environment.zip#env \
    generate-training-data-from-datasets.py

You may need to set some parameters for `spark-submit` depending on the cluster environment. For examples, you can inspect scripts [`run-data-generation-spark-cluster`](run-data-generation-spark-cluster) and [`run-model-training-spark-cluster`](run-model-training-spark-cluster).

### Output

The data generation process will create all the query and candidate datasets under `new_datasets_directory` (if `skip_dataset_creation=false`), as well as a training and test data files (`training-data-*` and `test-data-*`) that contains multiple JSON objects of the following format:

```
{
    "query_dataset": the id of the query dataset
    "target": the name of the target variable
    "candidate_dataset": the id of the candidate dataset
    "imputation_strategy": the missing value imputation strategy used after the join between the query and the candidate datasets, or "null" if inner join was applied instead
    "mark": indicates whether the data point represents a positively-constructed example or a negatively-constructed example ('n/a' if this information is not available)
    "mean_absolute_error": an array where the first and second values correspond to the mean absolute error before and after augmentation, respectively
    "mean_squared_error": an array where the first and second values correspond to the mean squared error before and after augmentation, respectively
    "median_absolute_error": an array where the first and second values correspond to the median absolute error before and after augmentation, respectively
    "r2_score": an array where the first and second values correspond to the R^2 score before and after augmentation, respectively
}
```

Note that one training and one test data files are generated for each regression algorithm chosen.

### Logs from Apache YARN

If you redirect the `stdout` of the job submission to a file, you can run the following script to capture the job's application id:

    $ python logs/capture-application-id.py <stdout file>

Given this id, you can retrieve its corresponding logs by running `yarn logs`:

    $ yarn logs -applicationId <application id>

For your convenience, the script [`run-spark-job-cluster`](run-spark-job-cluster) automatically runs the data generation process on the cluster and retrieves the corresponding logs:

    $ ./run-spark-job-cluster <run script> <output name>

where `<run script>` is the `spark-submit` script (e.g.: [`run-data-generation-spark-cluster`](run-data-generation-spark-cluster) or [`run-model-training-spark-cluster`](run-model-training-spark-cluster)) and `<output name>` is the desired name for the logs. This creates two files: `logs/<output name>.out`, which contains the `stdout` of the job submission, and `logs/<output name>.log`, with contains the logs.

### Generating Data Statistics

To generate some statistics about the generated data, you can run the script [`generate-stats-from-training-data.py`](generate-stats-from-training-data.py) using Spark. For instance, to run it locally:

    $ spark-submit \
    --deploy-mode client \
    --files .params.json \
    generate-stats-from-training-data.py

The output (part of `stdout`) can then be used by script [`generate-plots-from-stats.py`](generate-plots-from-stats.py) to create some plots. These plots are automatically saved under directory `plots/`, which will be created if it does not originally exists. Note that [`generate-plots-from-stats.py`](generate-plots-from-stats.py) uses [Matplotlib](https://matplotlib.org/) for the plots.

For your convenience, the scripts [`run-stats-client`](run-stats-client) and [`run-stats-cluster`](run-stats-cluster) automatically generate the statistics and create the plots for local and cluster executions, respectively. For instance, if you are running in a cluster:

    $ ./run-stats-cluster <output name>

where `<output name>` is the desired name for the plots.


## OpenML Datasets

If you want to download [OpenML](https://www.openml.org/) datasets to use them in the data generation process, check it [here](openml-datasets).
