.. -*- mode: rst -*-

svmlight-loader
===============

This is a fast and memory efficient loader for the svmlight / libsvm sparse data file format in Python.


Install
=======

To install for all users on Unix/Linux::

  python setup.py build
  sudo python setup.py install

API
====

This project includes a fast utility function, ``load_svmlight_format``,  to load
datasets in the svmlight / libsvm format. In this format, each line
takes the form ``<label> <feature-id>:<feature-value>
<feature-id>:<feature-value> ...``. This format is especially suitable for sparse datasets.
Scipy sparse CSR matrices are used for ``X`` and numpy arrays are used for ``y``.

You may load a dataset like this::

  >>> from svmlight_loader import load_svmlight_file
  >>> X_train, y_train = load_svmlight_file("/path/to/train_dataset.txt")


You may also load two datasets at once::

  >>> X_train, y_train, X_test, y_test = load_svmlight_file(
  ...     "/path/to/train_dataset.txt",
  ...     "/path/to/test_dataset.txt")

In this case, ``X_train`` and ``X_test`` are guaranteed to have the same number
of features. Another way to achieve the same result is to fix the number of
features::

  >>> X_test, y_test = load_svmlight_file(
  ...     "/path/to/test_dataset.txt", n_features=X_train.shape[1])

Public datasets
===============

Public datasets in svmlight / libsvm format available at http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/

License
=======

Simple BSD.

Authors
=======

Mathieu Blondel and Lars Buitinck



