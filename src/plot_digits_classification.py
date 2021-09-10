"""
================================
Recognizing hand-written digits
================================

This example shows how scikit-learn can be used to recognize images of
hand-written digits, from 0-9.
"""

print(__doc__)

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause

# Standard scientific Python imports
import matplotlib.pyplot as plt

import numpy as np

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# Importing rescale, resize, reshape
from skimage.transform import rescale, resize, downscale_local_mean 

###############################################################################
# Digits dataset
# --------------
#
# The digits dataset consists of 8x8
# pixel images of digits. The ``images`` attribute of the dataset stores
# 8x8 arrays of grayscale values for each image. We will use these arrays to
# visualize the first 4 images. The ``target`` attribute of the dataset stores
# the digit each image represents and this is included in the title of the 4
# plots below.
#
# Note: if we were working from image files (e.g., 'png' files), we would load
# them using :func:`matplotlib.pyplot.imread`.

digits = datasets.load_digits()

print("shape of data:", digits.images.shape)
print("shape of single image:", digits.images[0].shape, end="\n\n")


_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, label in zip(axes, digits.images, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    ax.set_title('Training: %i' % label)

###############################################################################
# Classification
# --------------
#
# To apply a classifier on this data, we need to flatten the images, turning
# each 2-D array of grayscale values from shape ``(8, 8)`` into shape
# ``(64,)``. Subsequently, the entire dataset will be of shape
# ``(n_samples, n_features)``, where ``n_samples`` is the number of images and
# ``n_features`` is the total number of pixels in each image.
#
# We can then split the data into train and test subsets and fit a support
# vector classifier on the train samples. The fitted classifier can
# subsequently be used to predict the value of the digit for the samples
# in the test subset.




def digitsClassifier(data, test_size, gamma=0.001):
    #print("\n\ndata shape:", data.shape)
    #print("train-test split is:", 1-test_size,":",test_size, "\n\n")

    # flatten the images
    n_samples = len(data)
    data = data.reshape((n_samples, -1))

    # Split data into 50% train and 50% test subsets
    X_train, X_test, y_train, y_test = train_test_split(
        data, digits.target[:len(data)], test_size=test_size, shuffle=False)

    # Create a classifier: a support vector classifier
    clf = svm.SVC(gamma=gamma)

    # Learn the digits on the train subset
    clf.fit(X_train, y_train)

    # Predict the value of the digit on the test subset
    predicted = clf.predict(X_test)
    a = round(accuracy_score(y_test, predicted), 4)
    p = round(precision_score(y_test, predicted, average='macro', zero_division=0), 4)
    r = round(recall_score(y_test, predicted, average='macro', zero_division=0), 4)
    f1 = round(f1_score(y_test, predicted, average='macro', zero_division=0), 4)

    return [a, p, r, f1]
    

# checking for different gamma values

data_org = digits.images
for gamma in [0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]:
    results_gamma = []
    print(f"for gamma = {gamma}")
    results_gamma.append(digitsClassifier(data_org, test_size=0.3, gamma=gamma))
# print("gamma results:", results_gamma)
    for r in results_gamma:
        print(f"a:{r[0]}  p:{r[1]}  r:{r[2]}  f1:{r[3]} \n")


results = []
data_org = digits.images
print("data_org:", data_org.shape)
# calling the function on original data
results.append(digitsClassifier(data_org, test_size=0.2))
results.append(digitsClassifier(data_org, test_size=0.3))
results.append(digitsClassifier(data_org, test_size=0.4))

# rescaling images 
image_rescaled = []
for img in data_org:
    image_rescaled.append(rescale(img, 0.9, anti_aliasing=False))
image_rescaled = np.array(image_rescaled)
print("\n\nimage_rescaled:", image_rescaled.shape)

results.append(digitsClassifier(image_rescaled, test_size=0.2))
results.append(digitsClassifier(image_rescaled, test_size=0.3))
results.append(digitsClassifier(image_rescaled, test_size=0.4))

image_rescaled = []
for img in data_org:
    image_rescaled.append(rescale(img, 0.75, anti_aliasing=False))
image_rescaled = np.array(image_rescaled)
print("\n\nimage_rescaled:", image_rescaled.shape)

results.append(digitsClassifier(image_rescaled, test_size=0.2))
results.append(digitsClassifier(image_rescaled, test_size=0.3))
results.append(digitsClassifier(image_rescaled, test_size=0.4))

image_rescaled = []
for img in data_org:
    image_rescaled.append(rescale(img, 0.6, anti_aliasing=False))
image_rescaled = np.array(image_rescaled)
print("\n\nimage_rescaled:", image_rescaled.shape)

results.append(digitsClassifier(image_rescaled, test_size=0.2))
results.append(digitsClassifier(image_rescaled, test_size=0.3))
results.append(digitsClassifier(image_rescaled, test_size=0.4))


# resizing images 
image_resized = []
for img in data_org:
    image_resized.append(resize(img, (data_org[0].shape[0] // 4, data_org[0].shape[1] // 4), anti_aliasing=True))
image_resized = np.array(image_resized)
print("\n\nimage_resized:", image_resized.shape)

results.append(digitsClassifier(image_resized, test_size=0.2))
results.append(digitsClassifier(image_resized, test_size=0.3))
results.append(digitsClassifier(image_resized, test_size=0.4))

image_resized = []
for img in data_org:
    image_resized.append(resize(img, (data_org[0].shape[0] // 2, data_org[0].shape[1] // 2), anti_aliasing=True))
image_resized = np.array(image_resized)
print("\n\nimage_resized:", image_resized.shape)

results.append(digitsClassifier(image_resized, test_size=0.2))
results.append(digitsClassifier(image_resized, test_size=0.3))
results.append(digitsClassifier(image_resized, test_size=0.4))


for r in results:
    print(f"{r[0]}     {r[1]}    {r[2]}    {r[3]}")



