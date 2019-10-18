# OpenML Datasets

## Requirements

* Python 3
* pandas
* [openml](https://pypi.org/project/openml/)

## Downloading Data

To download OpenML datasets for the data generation process, run the following:

    $ python retrieve-openml-datasets.py <output directory>

where `<output directory>` is the directory where all the datasets will be downloaded.

## Uploading to HDFS

If you need to upload the OpenML datasets to HDFS, you can run the following script:

    $ ./cp-to-hdfs <openml-directory> <hdfs-directory>

where `<openml-directory>` is the same as the previous `<output directory>`, and `<hdfs-directory>` is the directory on HDFS where the datasets should be saved to.