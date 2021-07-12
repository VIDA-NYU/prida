# PRIDA: Pruning Irrelevant Datasets for Data Augmentation

Let `Q` be an input (query) dataset, `t` a target variable from `Q`, and `M` a machine learning model that uses `Q` to predict `t`. Given a set `C` of datasets that can be used to augment `Q`, the goal of this project is to prune the candidate datasets that are unlikely to improve the performance of `M` through data augmentation. 

The main steps are:

1. **Find Candidate Datasets**. The first step is to, given `Q`, efficiently retrieve a set of candidate datasets that can be used to augment `Q`. For now, we focus on augmentation by joins. Efficient data structures and algorithms have been recently proposed to tackle this problem, such as [Lazo](https://github.com/mitdbg/lazo).
2. **Predict Performance Improvements from Candidate Augmentations**. The second step is to, given the set `C` of candidate datasets from step 1, predict whether these datasets are likely to improve `M` and prune accordingly. To do that, we create a metamodel that, for each candidate dataset `c` from `C`, classifies it as relevant or irrelevant for augmentation  *without having to do the augmentation or to re-train `M`*. 
3. **Generate Training Data**. To train, validate, and test our model from step 2, we need to generate training (ground-truth) data, composed of different `Q` and `C`, with their corresponding labels (relevant or irrelevant) after augmentation.

## Predicting Performance Improvements and Classifying Candidates

The code to predict performance improvement is available [here](improvement-prediction)

## Generating Training Data

To generate ground-truth data for training and testing, we use datasets from the D3M project and from OpenML. The main idea is to break each dataset into different query and candidate datasets, randomly choosing the columns. We also randomly remove records from query and candidate datasets, to avoid perfect joins.

The code to generate training data is available [here](data-generation).
