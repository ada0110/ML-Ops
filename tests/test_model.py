import sys
sys.path.extend([".", ".."])

import os
import numpy as np
from sklearn import datasets
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# import utils
from src.utils import preprocess, data_split, get_scores, digitsClassifier, save_model


def test_model_writing():
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

    path = "../models/test_save_model.pkl"
    save_model(clf, path)

    assert os.path.isfile(path) == True, f"model not saved at {path}!"


def test_small_data_overfit_checking():
    k = 50
    threshold = 0.9

    # data
    digits = datasets.load_digits()

    # take small sample
    data_org = digits.images[:k]
    target = digits.target[:k]

    # preprocess
    data = preprocess(data_org)

    # split
    x_train, x_test, x_val, y_train, y_test, y_val = data_split(data, target)
    # train
    clf = digitsClassifier(x_train, y_train)
     
    # get_scores returns list of [acc, precision, recall, f1]
    train_scores = get_scores(clf, x_train, y_train)

    print("train:", train_scores)

    assert train_scores[0] > threshold, "didn't overfit as per accuracy metric!"
    assert train_scores[3] > threshold, "didn't overfit as per f1 metric!"


# """
# # TODO: write  a test case to check if model is successfully getting created or not?
# def test_model_writing():

#     1. create some data

#     2. run_classification_experiment(data, expeted-model-file)

#     assert os.path.isfile(expected-model-file)


# # TODO: write a test case to check fitting on training -- litmus test.

# def test_small_data_overfit_checking():

#     1. create a small amount of data / (digits / subsampling)

#     2. train_metrics = run_classification_experiment(train=train, valid=train)

#     assert train_metrics['acc']  > some threshold

#     assert train_metrics['f1'] > some other threshold
# """

# def test_preprocess():
#     pass


# def test_data_split():
#     pass


# def test_get_scores():
#     pass

# # experiment within test case where model should be created at some place
# # method to save model
# # given training and validation --> write model onto disk
# # test case to check if model is gettibng created correctly

# # model should overfit on training data 
# # that is perform better than random classifier --> litmus test
# #  


# def test_model_writing():
#     # creating some data
#     x = np.random.rand(1000, 8)
#     y = np.random.randint(10, size=1000)
#     x_train, x_test, x_val, y_train, y_test, y_val = data_split(x, y)
#     clf = digitsClassifier(x_train, y_train)
#     # saving model
#     path = "/home/ada/codes/ML-Ops_Scikit/models/test.pkl"
#     save_model(clf, path)
#     assert os.path.isfile(path), "Model not saved"



