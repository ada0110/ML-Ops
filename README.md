# ML-Ops_Scikit from feature/plot branch

> To avoid folder name confusion, I have put code in `src` folder.


To run the Classification example follow below instructions:

1. Create a new conda environment  
```conda create --name new_env_name python=3.6```

2. Switch to this environment  
```conda activate new_env_name```

3. To install requirements, use conda. This will ensure that dependencies are resolved correctly.  
```conda install --file requirements.txt```

**Output**

```
================================
Recognizing hand-written digits
================================

This example shows how scikit-learn can be used to recognize images of
hand-written digits, from 0-9.

Classification report for classifier SVC(gamma=0.001):
              precision    recall  f1-score   support

           0       1.00      0.99      0.99        88
           1       0.99      0.97      0.98        91
           2       0.99      0.99      0.99        86
           3       0.98      0.87      0.92        91
           4       0.99      0.96      0.97        92
           5       0.95      0.97      0.96        91
           6       0.99      0.99      0.99        91
           7       0.96      0.99      0.97        89
           8       0.94      1.00      0.97        88
           9       0.93      0.98      0.95        92

    accuracy                           0.97       899
   macro avg       0.97      0.97      0.97       899
weighted avg       0.97      0.97      0.97       899


Confusion matrix:
[[87  0  0  0  1  0  0  0  0  0]
 [ 0 88  1  0  0  0  0  0  1  1]
 [ 0  0 85  1  0  0  0  0  0  0]
 [ 0  0  0 79  0  3  0  4  5  0]
 [ 0  0  0  0 88  0  0  0  0  4]
 [ 0  0  0  0  0 88  1  0  0  2]
 [ 0  1  0  0  0  0 90  0  0  0]
 [ 0  0  0  0  0  1  0 88  0  0]
 [ 0  0  0  0  0  0  0  0 88  0]
 [ 0  0  0  1  0  1  0  0  0 90]]
```

```
image size    train:test       accuracy   precision  recall    f1
  8*8          80:20            0.9583     0.9594    0.9586    0.9578
  8*8          70:30            0.9704     0.9709    0.9704    0.9703
  8*8          60:40            0.9652     0.9656    0.9658    0.9649
  
  7*7          80:20            0.9417     0.9427    0.9416    0.9405
  7*7          70:30            0.95       0.9504    0.9496    0.9495
  7*7          60:40            0.9555     0.9561    0.9557    0.9553
  
  6*6          80:20            0.9306     0.9351    0.9313    0.929
  6*6          70:30            0.9463     0.9473    0.946     0.9456
  6*6          60:40            0.9444     0.946     0.9445    0.944
  
  5*5          80:20            0.9306     0.9325    0.9304    0.9294
  5*5          70:30            0.9241     0.9244    0.9233    0.9231
  5*5          60:40            0.9305     0.9319    0.9302    0.9297  
  
  4*4          80:20            0.8667     0.8707    0.8657    0.8655     
  4*4          70:30            0.8537     0.857     0.8522    0.8496
  4*4          60:40            0.8387     0.8555    0.8392    0.8276
  
  2*2          80:20            0.4944     0.4479    0.489     0.4564
  2*2          70:30            0.4389     0.4196    0.4394    0.3825
  2*2          60:40            0.459      0.372     0.4558    0.3969
```
**Observations**
1. After resizing the images to lower dimensions, the numbers for accuracy, precision, recall and f1 are decreasing. This is due to loss in information.
2. With respect to train-test split, for 8*8, 7*7 and 6*6, the numbers are better for 70:30 split. In general, as we decrease training samples, numbers go down.
