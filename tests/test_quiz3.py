'''
Note: load the best svm_model from the disc -- do not train it during the test case.
1. Add one positive test case per class. For example, "def test_digit_correct_0" function tests if the prediction of an actual digit-0 sample indeed 0 or not, i.e. `assert prediction==0`. (Total of 10 such test cases)
2. [Bonus] Add a test case that checks that accuracy on each class is greater than a certain threshold. i.e. `assert acc_digit[0] > min_acc_req`
'''

import sys

import numpy
sys.path.extend([".", ".."])

# import utils
from src.utils import preprocess, data_split
import pickle
from sklearn import datasets


svm_best_model_path = '/home/ada/codes/ML-Ops_Scikit/models/best_model_svm.pkl'
decision_best_model_path = '/home/ada/codes/ML-Ops_Scikit/models/best_model_decision.pkl'

print("\nloading the svm_model...")
load_file = open(svm_best_model_path, "rb")
svm_model = pickle.load(load_file)

print("\nloading the decision_model...")
load_file = open(svm_best_model_path, "rb")
decision_model = pickle.load(load_file)

# data
digits = datasets.load_digits()
data_org = digits.images
target = digits.target

# preprocess
data = preprocess(data_org)

# split
x_train, x_test, x_val, y_train, y_test, y_val = data_split(data, target)
samples = []
targets = []

print(y_test[:10])

# making samples
for i in range(10):
    idx_i = [y_test==i]
    samples.append(x_test[idx_i][0])
    targets.append(i)

samples = numpy.array(samples)
print(f"len samples:{len(samples)}")
print(f"targets:{targets}")


# svm
def test_digit_correct_0():
    prediction = svm_model.predict(samples[0].reshape(1, -1))
    print(samples[0].reshape(1, -1))
    print(prediction)
    print(prediction[0], targets[0])
    assert prediction[0] == targets[0], f"Prediction incorrect"

def test_digit_correct_1():
    prediction = svm_model.predict(samples[1].reshape(1, -1))
    assert prediction[0] == targets[1], f"Prediction incorrect"

def test_digit_correct_2():
    prediction = svm_model.predict(samples[2].reshape(1, -1))
    assert prediction[0] == targets[2], f"Prediction incorrect"

def test_digit_correct_3():
    prediction = svm_model.predict(samples[3].reshape(1, -1))
    assert prediction[0] == targets[3], f"Prediction incorrect"

def test_digit_correct_4():
    prediction = svm_model.predict(samples[4].reshape(1, -1))
    assert prediction[0] == targets[4], f"Prediction incorrect"

def test_digit_correct_5():
    prediction = svm_model.predict(samples[5].reshape(1, -1))
    assert prediction[0] == targets[5], f"Prediction incorrect"

def test_digit_correct_6():
    prediction = svm_model.predict(samples[6].reshape(1, -1))
    assert prediction[0] == targets[6], f"Prediction incorrect"

def test_digit_correct_7():
    prediction = svm_model.predict(samples[7].reshape(1, -1))
    assert prediction[0] == targets[7], f"Prediction incorrect"

def test_digit_correct_8():
    prediction = svm_model.predict(samples[8].reshape(1, -1))
    assert prediction[0] == targets[8], f"Prediction incorrect"

def test_digit_correct_9():
    prediction = svm_model.predict(samples[9].reshape(1, -1))
    assert prediction[0] == targets[9], f"Prediction incorrect"



# decision
def test_decision_digit_correct_0():
    prediction = decision_model.predict(samples[0].reshape(1, -1))
    print(samples[0].reshape(1, -1))
    print(prediction)
    print(prediction[0], targets[0])
    assert prediction[0] == targets[0], f"Prediction incorrect"

def test_decision_digit_correct_1():
    prediction = decision_model.predict(samples[1].reshape(1, -1))
    assert prediction[0] == targets[1], f"Prediction incorrect"

def test_decision_digit_correct_2():
    prediction = decision_model.predict(samples[2].reshape(1, -1))
    assert prediction[0] == targets[2], f"Prediction incorrect"

def test_decision_digit_correct_3():
    prediction = decision_model.predict(samples[3].reshape(1, -1))
    assert prediction[0] == targets[3], f"Prediction incorrect"

def test_decision_digit_correct_4():
    prediction = decision_model.predict(samples[4].reshape(1, -1))
    assert prediction[0] == targets[4], f"Prediction incorrect"

def test_decision_digit_correct_5():
    prediction = decision_model.predict(samples[5].reshape(1, -1))
    assert prediction[0] == targets[5], f"Prediction incorrect"

def test_decision_digit_correct_6():
    prediction = decision_model.predict(samples[6].reshape(1, -1))
    assert prediction[0] == targets[6], f"Prediction incorrect"

def test_decision_digit_correct_7():
    prediction = decision_model.predict(samples[7].reshape(1, -1))
    assert prediction[0] == targets[7], f"Prediction incorrect"

def test_decision_digit_correct_8():
    prediction = decision_model.predict(samples[8].reshape(1, -1))
    assert prediction[0] == targets[8], f"Prediction incorrect"

def test_decision_digit_correct_9():
    prediction = decision_model.predict(samples[9].reshape(1, -1))
    assert prediction[0] == targets[9], f"Prediction incorrect"


