# Training Data Generation

## Requirements

* Python 3
* Numpy
* Pandas

## Generating Data

To generate training data, first, configure the file [`params.json`](params.json):

```
{
    "datasets_directory": directory of D3M datasets
    "output_directory": output directory where files are saved
    "training_data_file": output file that stores information about the training data
    "regression_algorithm": the regression algorithm to be used; choices are: "svm", "random forest", or "linear regression"
    "max_times_break_data_vertical": the maximum number of times that a dataset will be broken (vertically) into multiple data
    "max_times_records_removed": the maximum number of times that records will be removed from a dataset to derive new data
    "max_ratio_records_removed": the maximum ratio of records to be removed from a dataset to derive new data
}
```

Then, run the following:

    $ python generate-data-from-d3m-datasets.py
