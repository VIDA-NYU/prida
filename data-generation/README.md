# Training Data Generation

## Requirements

* Python 3
* Pandas

## Generating Data

To generate training data, first, configure the file [`params.json`](params.json):

```
{
    "datasets_directory": directory of D3M datasets
    "output_directory": output directory where files are saved
    "training_data_file": output file that stores information about the training data
    "regression_algorithm": the regression algorithm to be used; choices are: "svm", "random forest", or "linear regression"
}
```

Then, run the following:

    $ python generate-data-from-d3m-datasets.py
