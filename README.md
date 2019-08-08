# Learning to Augment: Ranking Datasets for Data Augmentation

Let `Q` be an input (query) dataset, `t` a target variable from `Q`, and `M` a machine learning model that uses `Q` to predict `t`. Given a set `C` of datasets that can be used to augment `Q`, the goal of this project is to rank these candidate datasets such that augmentations that potentially improve the performance of `M` are ranked higher.

The main steps are:

1. **Find Candidate Datasets**. The first step is to, given `Q`, efficiently retrieve a set of candidate datasets that can be used to augment `Q`. For now, we focus on augmentation by joins. Efficient data structures and algorithms have been recently proposed to tackle this problem, and we use Lazo (either the [original library](https://github.com/mitdbg/lazo) or the [index service](https://gitlab.com/ViDA-NYU/datamart/lazo-index-service)) for this work.
2. **Predict Performance Improvements from Candidate Augmentations**. The second step is to, given the set `C` of candidate datasets from step 1, rank these datasets based on how much they improve `M`. To do that, we use learning-to-rank and create a model that, for each candidate dataset `c` from `C`, predicts the performance improvement *without having to do the augmentation and re-train `M`*. 
3. **Generate Training Data**. To train, validate, and test our model from step 2, we need to generate training (ground-truth) data, composed of different `Q` and `C`, with their corresponding performance increases after augmentation.

## 1. Find Candidate Datasets

TBD

## 2. Predict Performance Improvements

TBD

## 3. Generate Training Data

TBD