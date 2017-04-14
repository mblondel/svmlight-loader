import numpy as np
import os

from numpy.testing import assert_equal, assert_array_equal
from nose.tools import raises

from svmlight_loader import (load_svmlight_file, load_svmlight_files,
                             dump_svmlight_file)
from sklearn.datasets import load_svmlight_file as sk_load_svmlight_file

currdir = os.path.dirname(os.path.abspath(__file__))
datafile = os.path.join(currdir, "data", "svmlight_classification.txt")
invalidfile = os.path.join(currdir, "data", "svmlight_invalid.txt")

qid_datafile         = os.path.join(currdir, "data", "svmlight_classification_qid.txt")

def test_load_svmlight_qid_file():
    X, y, c, q = load_svmlight_file(qid_datafile, comment=True, query_id=True)

    # test X's shape
    assert_equal(X.indptr.shape[0], 4)
    assert_equal(X.shape[0], 3)
    assert_equal(X.shape[1], 20)
    assert_equal(y.shape[0], 3)

    # test X's non-zero values
    for i, j, val in ((0, 1, 2.5), (0, 9, -5.2), (0, 14, 1.5),
                     (1, 4, 1.0), (1, 11, -3),
                     (2, 19, 27)):

        assert_equal(X[i, j], val)

    # tests X's zero values
    assert_equal(X[0, 2], 0)
    assert_equal(X[0, 4], 0)
    assert_equal(X[1, 7], 0)
    assert_equal(X[1, 15], 0)
    assert_equal(X[2, 17], 0)

    # test can change X's values
    X[0, 1] *= 2
    assert_equal(X[0, 1], 5)

    # test y
    assert_array_equal(y, [1, 2, 3])

    # test c
    assert_array_equal(c, [" an inline comment", "", ""])

    # test q
    assert_array_equal(q, [1, 37, 12])


def test_load_svmlight_file_empty_qid():
    X, y, c, q = load_svmlight_file(datafile, comment=True, query_id=True)

    # test X's shape
    assert_equal(X.indptr.shape[0], 4)
    assert_equal(X.shape[0], 3)
    assert_equal(X.shape[1], 20)
    assert_equal(y.shape[0], 3)

    # test X's non-zero values
    for i, j, val in ((0, 1, 2.5), (0, 9, -5.2), (0, 14, 1.5),
                     (1, 4, 1.0), (1, 11, -3),
                     (2, 19, 27)):

        assert_equal(X[i, j], val)

    # tests X's zero values
    assert_equal(X[0, 2], 0)
    assert_equal(X[0, 4], 0)
    assert_equal(X[1, 7], 0)
    assert_equal(X[1, 15], 0)
    assert_equal(X[2, 17], 0)

    # test can change X's values
    X[0, 1] *= 2
    assert_equal(X[0, 1], 5)

    # test y
    assert_array_equal(y, [1, 2, 3])

    # test c
    assert_array_equal(c, [" an inline comment", "", ""])

    # test q
    assert_equal(q.shape[0], 0)

def test_load_svmlight_file():
    X, y = load_svmlight_file(datafile)

    # test X's shape
    assert_equal(X.indptr.shape[0], 4)
    assert_equal(X.shape[0], 3)
    assert_equal(X.shape[1], 20)
    assert_equal(y.shape[0], 3)

    # test X's non-zero values
    for i, j, val in ((0, 1, 2.5), (0, 9, -5.2), (0, 14, 1.5),
                     (1, 4, 1.0), (1, 11, -3),
                     (2, 19, 27)):

        assert_equal(X[i, j], val)

    # tests X's zero values
    assert_equal(X[0, 2], 0)
    assert_equal(X[0, 4], 0)
    assert_equal(X[1, 7], 0)
    assert_equal(X[1, 15], 0)
    assert_equal(X[2, 17], 0)

    # test can change X's values
    X[0, 1] *= 2
    assert_equal(X[0, 1], 5)

    # test y
    assert_array_equal(y, [1, 2, 3])

def test_load_svmlight_files_comment_qid():
    X_train, y_train, c_train, q_train, X_test, y_test, c_test, q_test = load_svmlight_files([datafile] * 2,
                                                           dtype=np.float32, comment=True, query_id=True)
    assert_array_equal(X_train.toarray(), X_test.toarray())
    assert_array_equal(y_train, y_test)
    assert_equal(X_train.dtype, np.float32)
    assert_equal(X_test.dtype, np.float32)

    X1, y1, c1, q1, X2, y2, c2, q2, X3, y3, c3, q3 = load_svmlight_files([datafile] * 3,
                                                 dtype=np.float64, comment=True, query_id=True)
    assert_equal(X1.dtype, X2.dtype)
    assert_equal(X2.dtype, X3.dtype)
    assert_equal(X3.dtype, np.float64)

def test_load_svmlight_files():
    X_train, y_train, X_test, y_test = load_svmlight_files([datafile] * 2,
                                                           dtype=np.float32)
    assert_array_equal(X_train.toarray(), X_test.toarray())
    assert_array_equal(y_train, y_test)
    assert_equal(X_train.dtype, np.float32)
    assert_equal(X_test.dtype, np.float32)

    X1, y1, X2, y2, X3, y3 = load_svmlight_files([datafile] * 3,
                                                 dtype=np.float64)
    assert_equal(X1.dtype, X2.dtype)
    assert_equal(X2.dtype, X3.dtype)
    assert_equal(X3.dtype, np.float64)



def test_load_svmlight_file_n_features():
    X, y = load_svmlight_file(datafile, n_features=14)

    # test X'shape
    assert_equal(X.indptr.shape[0], 4)
    assert_equal(X.shape[0], 3)
    assert_equal(X.shape[1], 14)

    # test X's non-zero values
    for i, j, val in ((0, 1, 2.5), (0, 9, -5.2),
                     (1, 4, 1.0), (1, 11, -3)):

        assert_equal(X[i, j], val)


@raises(ValueError)
def test_load_invalid_file():
    load_svmlight_file(invalidfile)


@raises(ValueError)
def test_load_invalid_file2():
    load_svmlight_files([datafile, invalidfile, datafile])


@raises(TypeError)
def test_not_a_filename():
    load_svmlight_file(1)


@raises(IOError)
def test_invalid_filename():
    load_svmlight_file("trou pic nic douille")


def test_dump():
    try:
        Xs, y, = load_svmlight_file(datafile)
        tmpfile = "tmp_dump.txt"
        dump_svmlight_file(Xs, y, tmpfile, zero_based=False)
        X2, y2 = sk_load_svmlight_file(tmpfile)
        assert_array_equal(Xs.toarray(), X2.toarray())
        assert_array_equal(y, y2)
    finally:
        os.remove(tmpfile)

def test_dump_qid():
    try:
        Xs, y, q = load_svmlight_file(qid_datafile, query_id=True)
        tmpfile = "tmp_dump.txt"
        dump_svmlight_file(Xs, y, tmpfile, query_id=list(q), zero_based=False)
        X2, y2, q2 = sk_load_svmlight_file(tmpfile, query_id=True)
        assert_array_equal(Xs.toarray(), X2.toarray())
        assert_array_equal(y, y2)
        assert_array_equal(q, q2)
    finally:
        os.remove(tmpfile)

