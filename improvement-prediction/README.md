# Prediction of Performance Improvement

## Requirements

* [Python 3](https://www.python.org/)
* [HdfsCLI](https://hdfscli.readthedocs.io/en/latest/)
* [NumPy](https://numpy.org/)
* [pandas](https://pandas.pydata.org/)
* [scikit-learn](https://scikit-learn.org/stable/)
* [Apache Spark 2.3.0](https://spark.apache.org/)


## Predicting Improvement

To predict the improvement in the prediction of a target variable after performing a certain data augmentation, we first have to configure a parameter file, e.g., [`params.json`](params.json):

```
{
  "learning_data_training": reference to file that stores information about the learning task for training
  "learning_data_test": reference to file that stores information about the learning task for test
  "id_to_dataset_training": reference to file that stores all the datasets used for training
  "id_to_dataset_test": reference to file that stores all the datasets used for test
  "augmentation_learning_data_filename": reference to file that stores all features, and improvements obtained via data augmentation according to different metrics 
  "cluster": false,
  "hdfs_address": "http://gray01.poly.edu:50070",
  "hdfs_user": "fsc234"
}
```

### Generating features and relative prediction improvements

File `learning_data_file` is composed of JSON objects in the format:

```
{
    "query_dataset": the relative path for the query dataset
    "target": the name of the target variable
    "candidate_dataset": the relative path for the candidate dataset
    "imputation_strategy": the missing value imputation strategy used after the join between the query and the candidate datasets, or "null" if inner join was applied instead
    "mean_absolute_error": an array where the first and second values correspond to the mean absolute error before and after augmentation, respectively
    "mean_squared_error": an array where the first and second values correspond to the mean squared error before and after augmentation, respectively
    "median_absolute_error": an array where the first and second values correspond to the median absolute error before and after augmentation, respectively
    "r2_score": an array where the first and second values correspond to the R^2 score before and after augmentation, respectively
}
```

#### Client Mode

To generate features and actual prediction improvements associated to each augmentation instance (JSON object) in `learning_data_file`, run the following locally:

    $ if test -f dependencies.zip; then rm dependencies.zip; fi; zip -r dependencies.zip *.py util/; spark-submit \
			--deploy-mode client \
			--executor-memory 4G \
			--archives dependencies.zip \
			--files params.json generate_data_for_augmentation_learning_spark.py

You may need to set some parameters for `spark-submit` depending on your environment. For an example, you can inspect the script [`run-spark-client`](run-spark-client).

#### Cluster Mode (Apache YARN)

The easiest way to run the generation of features and prediction improvements in a cluster is by using [Anaconda](https://www.anaconda.com/) to package the python dependencies. First,
install Anaconda and initialize it by using the `conda init` command. Then, run the following:

    $ conda create -y -n feature-and-target-generation -c conda-forge python=3.6.9 numpy pandas scikit-learn python-hdfs 
    $ cd <env_dir>
    $ zip -r <feature-and-target-generation-dir>/feature-and-target-generation-environment.zip feature-and-target-generation/

where `<env_dir>` is the location for Anaconda environments (e.g., <path_to_anaconda3>/anaconda3/envs/), and `<feature-and-target-generation-dir>` is the location for this directory. The
`feature-and-target-generation-environment.zip` file will contain all the packages necessary to run the feature and target generation script, making sure that all of the cluster nodes have
access to the dependencies. To submit the feature and target generation job, run the following:

    $ cd <feature-and-target-generation-dir>
    $ if test -f dependencies.zip; then rm dependencies.zip; fi; zip -r dependencies.zip *.py util/; spark2-submit \
		--deploy-mode cluster \
		--master yarn \
		--conf spark.yarn.appMasterEnv.PYSPARK_PYTHON=./env/feature-and-target-generation/bin/python \
		--archives data/features-and-targets-environment.zip#env \
		--py-files dependencies.zip --files params.json \
		--driver-memory 6GB \
		--conf spark.dynamicAllocation.enabled=false \
		--conf spark.driver.memoryOverhead=1G \
		--conf spark.executor.memoryOverhead=1G \
		--num-executors 150 \
		--executor-cores 5 \
		--executor-memory 10G \
		--conf spark.network.timeout=12000s \
		generate_data_for_augmentation_learning_spark.py
		
You may need to set some parameters for `spark-submit` depending on the cluster environment. For examples, you can inspect script [`run-spark-cluster`](run-cluster).

#### Output (`augmentation_learning_data_filename`) 

Both the client and cluster mode executions generate file `augmentation_learning_data_filename`, which contains one line for each JSON object in `learning_data_file`. Each line contains features and relative
prediction improvements directly derived from the JSON object, and is in the format:

<query_filename, target_name, candidate_filename, number_of_columns_in_query_dataset, number_of_rows_in_query_dataset, row_to_column_ratio_in_query_dataset, number_of_numerical_columns_in_query_dataset,
means_of_numerical_columns_in_query_dataset, percentages_of_missing_values_in_query_dataset, outlier_percentages_of_numerical_columns_in_query_dataset, skewness_of_numerical_columns_in_query_dataset,
get_kurtosis_of_numerical_columns_in_query_dataset, number_of_unique_values_of_numerical_columns_in_query_dataset, entropy_levels_of_float_columns_in_query_dataset, entropy_levels_of_integer_columns_in_query_dataset,
number_of_columns_in_candidate_dataset, number_of_rows_in_candidate_dataset, row_to_column_ratio_in_candidate_dataset, number_of_numerical_columns_in_candidate_dataset, means_of_numerical_columns_in_candidate_dataset,
percentages_of_missing_values_in_candidate_dataset, outlier_percentages_of_numerical_columns_in_candidate_dataset, skewness_of_numerical_columns_in_candidate_dataset, get_kurtosis_of_numerical_columns_in_candidate_dataset,
number_of_unique_values_of_numerical_columns_in_candidate_dataset, entropy_levels_of_float_columns_in_candidate_dataset, entropy_levels_of_integer_columns_in_candidate_dataset,
max_pearson_in_modulus_in_joined_dataset, max_spearman_in_modulus_in_joined_dataset, max_kendalltau_in_modulus_in_joined_dataset, max_covariance_in_modulus_in_joined_dataset, max_mutual_info_in_modulus_in_joined_dataset,
max_pearson_in_modulus_with_target_in_joined_dataset, max_spearman_in_modulus_with_target_in_joined_dataset, max_kendalltau_in_modulus_with_target_in_joined_dataset, max_covariance_in_modulus_with_target_in_joined_dataset,
max_mutual_info_in_modulus_with_target_in_joined_dataset, max_pearson_in_modulus_with_target_in_query_dataset, max_spearman_in_modulus_with_target_in_query_dataset, max_kendalltau_in_modulus_with_target_in_query_dataset,
max_covariance_in_modulus_with_target_in_query_dataset, max_mutual_info_in_modulus_with_target_in_query_dataset, max_pearson_in_modulus_with_target_in_candidate_dataset, max_spearman_in_modulus_with_target_in_candidate_dataset,
max_kendalltau_in_modulus_with_target_in_candidate_dataset, max_covariance_in_modulus_with_target_in_candidate_dataset, max_mutual_info_in_modulus_with_target_in_candidate_dataset,
max_pearson_difference_wrt_target_between_query_and_candidate_datasets, ratio_of_rows_between_query_and_candidate_dataset, decrease_in_mean_absolute_error_predicting_target, decrease_in_mean_squared_error_predicting_target,
decrease_in_median_absolute_error_predicting_target, relative_gain_in_r2_score_predicting_target>


### Generating machine learning models for improvement prediction

The data in `augmentation_learning_data_filename` is used to create models that learn to discern between good and bad data augmentations. These models use the computed
features, or a subset of them, and predict values for the following: decrease_in_mean_absolute_error_predicting_target, decrease_in_mean_squared_error_predicting_target,
decrease_in_median_absolute_error_predicting_target, or relative_gain_in_r2_score_predicting_target. To build and run such models, do:

    $ python learn_to_augment.py

Parameter `n_splits` indicates how many folds are used in the cross-validation, through which different models can be evaluated and compared. ** TODO **
This script generates `output_filename`, a file that stores real performance improvements, such as relative_gain_in_r2_score_predicting_target, and their corresponding estimated
value for each test instance. Each line in the file is in the format: 

<query dataset filename, candidate dataset filename, target variable name, real_decrease_in_mean_absolute_error_predicting_target, predicted_decrease_in_mean_absolute_error_predicting_target,
real_decrease_in_mean_squared_error_predicting_target, predicted_decrease_in_mean_squared_error_predicting_target, real_decrease_in_median_absolute_error_predicting_target,
predicted_decrease_in_median_absolute_error_predicting_target, real_relative_gain_in_r2_score_predicting_target, predicted_relative_gain_in_r2_score_predicting_target>

### Recommending candidate datasets for augmentation

** TODO **

