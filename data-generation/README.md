# Generation of Training Data

## Requirements

* Python 3
* NumPy
* pandas
* scikit-learn

## Generating Data

To generate training data, first, configure the file [`params.json`](params.json):

```
{
    "datasets_directory": directory of D3M datasets
    "output_directory": output directory where files are saved
    "training_data_file": output file that stores information about the training data
    "regression_algorithm": the regression algorithm to be used; for now, the only available choices are "random forest" and "linear"
    "max_times_break_data_vertical": the maximum number of times that a dataset will be broken (vertically) into multiple data
    "max_times_records_removed": the maximum number of times that records will be removed from a dataset to derive new data
    "max_ratio_records_removed": the maximum ratio of records to be removed from a dataset to derive new data
}
```

Then, run the following:

    $ python generate-data-from-d3m-datasets.py

The script will generate all the query and candidate datasets under `output_directory`, and the file `training_data_file` will contain lines of the following format:

    <query dataset, candidate dataset, score before augmentation, score after augmentation>

The performance score is computed using the `R^2` regression function score.