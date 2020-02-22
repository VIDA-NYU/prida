# Error Analysis

The scripts in this folder aim to analyze false positives and false negatives present in the use cases.

## 1. Generation of Column 'eval'

Use script [`generate-column-eval.py`](generate-column-eval.py) to derive a copy of a use case dataset with added column 'eval', 
indicating, for each instance, whether it is a false positive (FP), true positive (TP), false negative (FN), or true negative (TN) according to a
given model. You can execute this script as follows:

    $ python generate-column-eval.py <use case dataset> <classifier-model>

Note that parameter <classifier-model> should be compatible with Python library pickle.

## 2. Generation of Feature an Target Histograms

Use script [`histogram-generation.py`](histogram-generation.py) to generate histograms for every feature and target, contrasting distributions for
false positives, false negatives, true positives, and true negatives. You can execute this script as follows:

    $ python histogram-generation.py <use case dataset with column 'eval'>

Note that this script depends on the existence of column 'eval', so make sure that you executed script [`generate-column-eval.py`](generate-column-eval.py)
as described in step 1.