# Generation of Training Data

## Requirements

* [Python 3](https://www.python.org/)
* [HdfsCLI](https://hdfscli.readthedocs.io/en/latest/)
* [NumPy](https://numpy.org/)
* [pandas](https://pandas.pydata.org/)
* [scikit-learn](https://scikit-learn.org/stable/)
* [XGBoost](https://xgboost.readthedocs.io/en/latest/python/python_intro.html)
* [Apache Spark 2.3.0](https://spark.apache.org/)

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
    "skip_dataset_creation": boolean that indicates whether the generation of query and candidate datasets should be skipped or not; if this step is skipped, only the scores before and after the augmentation are generated
    "candidate_single_column": boolean that indicates whether the candidate datasets should have a single column (in addition to the key column) or not
    "regression_algorithm": the regression algorithm to be used; the available options are "random forest", "linear", "sgd", and "xgboost"
    "inner_join": boolean that indicates whether the join applied between query and candidate datasets is of type inner or not; if false, a left join is applied and a series of univariate value imputation strategies are applied to take care of any missing values, with the one that generates the model with the smallest mean absolute error being chosen at last
    "min_number_records": the minimum number of records that a query or candidate dataset should have
    "max_times_break_data_vertical": the maximum number of times that a dataset will be broken (vertically) into multiple data
    "max_times_records_removed": the maximum number of times that records will be removed from a dataset to derive new data
    "max_ratio_records_removed": the maximum ratio of records to be removed from a dataset to derive new data
}
```

### Client Mode

To run the data generation process locally, run the following:

    $ spark-submit \
    --deploy-mode client \
    --files .params.json \
    generate-training-data-from-datasets.py

You may need to set some parameters for `spark-submit` depending on your environment. For an examples, you can inspect the script [`run-spark-client`](run-spark-client).

### Cluster Mode (Apache YARN)

The easiest way to run the data generation in a cluster is by using [Anaconda](https://www.anaconda.com/) to package the python dependencies. First, install Anaconda and initialize it by using the `conda init` command. Then, run the following:

    $ conda create -y -n data-generation -c conda-forge python=3.6.9 numpy pandas scikit-learn python-hdfs xgboost
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

You may need to set some parameters for `spark-submit` depending on the cluster environment. For an examples, you can inspect the script [`run-spark-cluster`](run-spark-cluster).

### Output

The data generation process will create all the query and candidate datasets under `new_datasets_directory` (if `skip_dataset_creation=false`), as well as training data files that contain lines of the following format:

    <query dataset, target variable name, candidate dataset, mean absolute error before augmentation, mean absolute error after augmentation, mean squared error before augmentation, mean squared error after augmentation, median absolute error before augmentation, median absolute error after augmentation, R^2 score before augmentation, R^2 score after augmentation>

## OpenML Datasets

If you want to download [OpenML](https://www.openml.org/) datasets to use them in the data generation process, check it [here](openml-datasets).
