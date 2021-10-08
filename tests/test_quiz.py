import sys

from scipy.sparse import data
sys.path.extend([".", ".."])

import os
import numpy as np
from sklearn import datasets
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# import utils
from src.utils import load_model, preprocess, data_split, get_scores, digitsClassifier, save_model


'''
Write test cases for the create_split
function
(the function that splits the data into train, test, and validation
sets) -- this function will be there in your codebase if you have
followed the code-refactoring lab session. Check for number of samples
in each split, and sum of samples in each split should come to the total number of samples. (so, 4 assert statements each in following two test cases)


1. if you give n=100 samples, and provide train:test:valid split as
70:20:10, there should be 70,20, and 10 samples in the train, tests, and
validation splits returned by create_split function.


2. given n=9 samples, with 70:20:10 split, the train should contain 6, 2, 1 samples respectively.
'''

def test_data_split_100():
    n = 100
    # split sizes
    train_size=70
    test_size=20 
    val_size=10

    # data
    digits = datasets.load_digits()

    data_org = digits.images[:n]
    target = digits.target[:n]
    data = preprocess(data_org)

    # split
    x_train, x_test, x_val, y_train, y_test, y_val = data_split(data, target, train_size=train_size, test_size=test_size, val_size=val_size)

    print(len(x_train), len(x_test), len(x_val))
    assert len(x_train) == train_size, "train size not correct"
    assert len(x_test) == test_size, "test size not correct"
    assert len(x_val) == val_size, "val size not correct"
    assert len(x_train)+len(x_test)+len(x_val) == n, "total size not correct"


def test_data_split_9():
    n = 9
    # split sizes
    train_size=6
    test_size=2 
    val_size=1

    # data
    digits = datasets.load_digits()

    # take small sample
    data_org = digits.images[:n]
    target = digits.target[:n]
    data = preprocess(data_org)
    
    # split
    x_train, x_test, x_val, y_train, y_test, y_val = data_split(data, target, train_size=train_size, test_size=test_size, val_size=val_size)

    print(len(x_train), len(x_test), len(x_val))
    assert len(x_train) == train_size, "train size not correct"
    assert len(x_test) == test_size, "test size not correct"
    assert len(x_val) == val_size, "val size not correct"
    assert len(x_train)+len(x_test)+len(x_val) == n, "total size not correct"


# bonus components
def test_data_label_size():
    n = 100
    # split sizes
    train_size=70
    test_size=20
    val_size=10

    # data
    digits = datasets.load_digits()

    # take small sample
    data_org = digits.images[:n]
    target = digits.target[:n]
    data = preprocess(data_org)
    
    # split
    x_train, x_test, x_val, y_train, y_test, y_val = data_split(data, target, train_size=train_size, test_size=test_size, val_size=val_size)

    assert len(x_train) == len(y_train), "train data, label size not correct"
    assert len(x_test) == len(y_test), "test data, label size not correct"
    assert len(x_val) == len(y_val), "val data, label size not correct"
    assert len(x_train)+len(x_test)+len(x_val) == len(y_train)+len(y_test)+len(y_val), "total size of data, label not correct"


def test_model_save_load():
    # data
    digits = datasets.load_digits()
    data_org = digits.images
    target = digits.target

    # preprocess
    data = preprocess(data_org)

    # split
    x_train, x_test, x_val, y_train, y_test, y_val = data_split(data, target)
    # train
    clf = digitsClassifier(x_train, y_train)

    # saving model
    path = "../models/test_save_model.pkl"
    save_model(clf, path)

    # laoding model
    loaded_model = load_model(path)
    print(loaded_model.support_vectors_)
    print(clf.support_vectors_)

    assert np.allclose(loaded_model.support_vectors_, clf.support_vectors_), f"saved model is not same as original!"

