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



--------------



## 4. Generating Learning Features

Use the [`generate-features-for-datamart-datasets` notebook](datamart-data/generate-features-for-datamart-datasets.ipynb) to generate the learning features from all the possible query-candidate pairs.

## 5. Predicting Gains

Use the [`predict-datamart-use-cases` notebook](datamart-data/predict-datamart-use-cases.ipynb) to generate the predictions for each possible query-candidate pair and the corresponding metrics (e.g.: precision and recall).