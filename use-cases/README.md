# Use Cases

## NYC Taxi and Vehicle Collisions Problem

The regression problem uses the number of taxi trips in NYC to predict the number of vehicle collisions in the city.

## NYC Taxi Demand Problem

The regression problem uses temporal features to predict the number of taxi trips in NYC.

## Poverty Estimation Problem

The task is to estimate the number of people living in poverty in 2016 across counties in the US using population estimates.

## College Debt Problem

The problem is to predict the median debt-to-earnings ratio of colleges across the US. The supplied dataset consists of a minimal version of the College Scorecards dataset, which make it easier for students to search for a college that is a good fit for them.

# Experiments with Use Cases

The Jupyter notebooks used for this experiment might have some hardcoded paths, so please change these paths accordingly when re-running the experiments.

## 1. Auctus Deployment

Make sure you have a local deployment of [Auctus](https://gitlab.com/ViDA-NYU/datamart/datamart), our dataset search engine tailored for data augmentation. For these experiments, we are using the [`dataset-recommendation` branch](https://gitlab.com/ViDA-NYU/datamart/datamart/-/tree/dataset-recommendation). Instruction on how to deploy Auctus locally are available [here](https://gitlab.com/ViDA-NYU/datamart/datamart/-/blob/dataset-recommendation/README.md#local-deployment-development-setup).

## 2. Importing Metadata to Auctus

Use the [`import-datasets-to-datamart` notebook](datamart-data/import-datasets-to-datamart.ipynb) to download and import metadata to Auctus.

## 3. Downloading Candidate Datasets and Computing Model Improvement

Use the [`download-datamart-datasets-joinable-with-query-datasets` notebook](datamart-data/download-datamart-datasets-joinable-with-query-datasets.ipynb) to download all the potential candidate datasets for the use cases. Auctus will automatically take care of the join operations. The datasets need to be downloaded so that the learning features can be computed. In addition, this notebook also computes the real model improvement so we can have the ground truth data.

## 4. Uploading Data to HDFS

First, we need to replace some candidate dataset names with ids, to avoid issues with HDFS:

    $ python change-data-to-id.py companion-datasets college-debt-datamart-records taxi-vehicle-collision-datamart-records poverty-estimation-datamart-records

Then, use the script [`cp-to-hdfs.py`](datamart-data/cp-to-hdfs.py) to copy all the files to HDFS:

    $ nohup python cp-to-hdfs.py <hdfs-directory> > cp-to-hdfs.out 2>&1 &

where `<hdfs-directory>` is the directory on HDFS where the datasets should be saved to.

## 5. Generating Learning Features

### 5.1. Transforming Data

We run a PySpark job to save the data in the expected format for learning features. First, copy the file [`params.json`](datamart-data/params.json), name it as `.params.json`, and configure it appropriately. The structure of this file is the following:

```
{
    "datasets_directory": directory of original datasets, i.e., <hdfs-directory>
    "new_datasets_directory": output directory where the query and the candidate datasets will be saved
    "hdfs_address": the address (host and port) for the distributed file system
    "hdfs_user": the username for the distributed file system
    "training_records": the file that contains all the training records for learning
}
```

Then, using [Anaconda](https://www.anaconda.com/), run the following to package all the required dependencies:

    $ conda create -y -n use-cases-data-generation -c conda-forge python=3.6.9 numpy pandas scikit-learn scipy python-hdfs xgboost
    $ cd <env_dir>
    $ zip -r <use-cases-dir>/use-cases-data-generation-environment.zip use-cases-data-generation/

where `<env_dir>` is the location for Anaconda environments, and `<use-cases-dir>` is the location for this directory. The `use-cases-data-generation-environment.zip` file will contain all the packages necessary to run the script, making sure that all of the cluster nodes have access to the dependencies. To submit the job, run the following:

    $ cd <use-cases-dir>
    $ spark-submit \
    --deploy-mode cluster \
    --master yarn \
    --files .params.json \
    --conf spark.yarn.appMasterEnv.PYSPARK_PYTHON=./env/use-cases-data-generation/bin/python \
    --archives use-cases-data-generation-environment.zip#env \
    split-data-for-use-cases.py

You may need to set some parameters for `spark-submit` depending on the cluster environment. For examples, you can inspect scripts [`run-split-data-for-use-cases`](datamart-data/run-split-data-for-use-cases).

### 5.2. Generating Features

TBD

## 6. Predicting Gains

Use the [`predict-datamart-use-cases` notebook](datamart-data/predict-datamart-use-cases.ipynb) to generate the predictions for each possible query-candidate pair and the corresponding metrics (e.g.: precision and recall).