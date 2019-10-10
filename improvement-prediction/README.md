# Prediction of Performance Improvement

## Requirements

* Python 3
* NumPy
* pandas
* scikit-learn

## Predicting Improvement

To predict performance improvement, first, configure the file [`params.json`](params.json):

```
{
    "learning_data_filename": reference to file that stores information about the learning task
		"augmentation_learning_data_filename": reference to file that stores all features, and relative `R^2` score improvement, derived from `learning_data_file` 
		"n_splits": number of folds for cross-validation when evaluating the learning task
		"output_filename": output filename where all predictions, along with original lines in learning_data_file, are saved
}
```

Then, run the following to derive data features from `learning_data_file`:

    $ python generate-data-for-augmentation-learning.py

Each line in `learning_data_file` is in the format

<query dataset filename, target attribute name, candidate dataset filename, `R^2` score before augmentation, `R^2` score after augmentation>

and each line in `augmentation_learning_data_filename` is in the format

<feature_1, ..., feature_N, relative `R^2` score improvement>

Note that `learning_data_file` and `augmentation_learning_data_filename` have the same number of lines (instances for learning), and that the lines in 
`augmentation_learning_data_filename` are the output of generate-data-for-augmentation-learning.py

Then, run the following to learn how to augment, creating a model that learns to discern between good and bad data augmentations: 

    $ python learn-to-augment.py
 
Each line in `output_filename` is in the format

<query dataset filename, candidate dataset filename, target attribute name, true relative `R^2` score improvement, predicted relative `R^2` score improvement>

where the true relative `R^2` score improvement obtained by augmenting the initial query dataset with a candidate dataset is computed as

true relative `R^2` score improvement = (`R^2` score after augmentation - `R^2` score before augmentation)/|`R^2` score before augmentation|
