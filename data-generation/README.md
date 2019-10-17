# Generation of Training Data

## Requirements

* Python 3
* HdfsCLI
* NumPy
* pandas
* scikit-learn
* Apache Spark

## Generating Data

The data generation process is done using PySpark. First, copy the file [`params.json`](params.json), name it as `.params.json`, and configure it appropriately. The structure of this file is the following:

```
{
    "original_datasets_directory": directory of original datasets
    "new_datasets_directory": output directory where the query and the candidate datasets will be saved
    "cluster": boolean that indicates whether the data generation process will be run in cluster or not
    "hdfs_address": the address (host and port) for the distributed file system; only used if the data generation will be run in a cluster
    "hdfs_user": the username for the distributed file system; only used if the data generation will be run in a cluster
    "ignore_first_attribute": boolean that indicates whether the first attribute of every dataset should be ignored (e.g.: the d3mIndex attribute)
    "skip_dataset_creation": boolean that indicates whether the generation of query and candidate datasets should be skipped or not; if this step is skipped, only the scores before and after the augmentation are generated
    "regression_algorithm": the regression algorithm to be used; for now, the only available choices are "random forest", "linear", and "sgd"
    "min_number_records": the minimum number of records that a query or candidate dataset should have
    "max_times_break_data_vertical": the maximum number of times that a dataset will be broken (vertically) into multiple data
    "max_times_records_removed": the maximum number of times that records will be removed from a dataset to derive new data
    "max_ratio_records_removed": the maximum ratio of records to be removed from a dataset to derive new data
}
```

Then, run the following:

    $ spark-submit --files .params.json generate-training-data-from-datasets.py

You may need to set some parameters for `spark-submit` depending on your environment and whether you are running it locally or in a cluster. For examples, you can inspect the scripts [`run-spark-client`](run-spark-client) and [`run-spark-cluster`](run-spark-cluster).

This process will generate all the query and candidate datasets under `new_datasets_directory` (if `skip_dataset_creation=false`), as well as training data files that contain lines of the following format:

    <query dataset, target variable name, candidate dataset, score before augmentation, score after augmentation>

The performance score is computed using the `R^2` regression function score.

## OpenML Datasets

If you want to download [OpenML](https://www.openml.org/) datasets to use them in the data generation process, check it [here](openml-datasets).