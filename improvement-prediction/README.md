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
		"n_splits": number of folds for cross-validation when evaluating the learning task
		"output_filename": output filename where all predictions, along with original lines in learning_data_file, are saved
}
```

Then, run the following:

    $ python predict-performance-improvement.py

Each line in `learning_data_file` is in the format

<query dataset filename, candidate dataset filename, target attribute name, `R^2` score before augmentation, `R^2` score after augmentation>

and each line in `output_filename` is in the format

<query dataset filename, candidate dataset filename, target attribute name, true relative `R^2` score improvement, predicted relative `R^2` score improvement>

where the true relative `R^2` score improvement obtained by augmenting the initial query dataset with a candidate dataset is computed as

true relative `R^2` score improvement = (`R^2` score after augmentation - `R^2` score before augmentation)/`R^2` score before augmentation
