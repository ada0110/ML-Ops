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
  7*7          80:20            0.0926     0.1125    0.0926    0.0845
  7*7          70:30            0.0926     0.0754    0.0936    0.0786
  7*7          60:40            0.1005     0.0812    0.1004    0.0866
  6*6          80:20            0.0963     0.0999    0.0955    0.0903
  6*6          70:30            0.1086     0.1242    0.1084    0.1066
  6*6          60:40            0.0926     0.0972    0.0924    0.0855
  5*5          80:20            0.0926     0.0831    0.0949    0.0801
  5*5          70:30            0.0802     0.0787    0.083     0.0709
  5*5          60:40            0.0972     0.1018    0.0972    0.086
  2*8          80:20            0.0667     0.0216    0.0698    0.0315
  2*8          70:30            0.0963     0.0096    0.1       0.0176
  2*8          60:40            0.0944     0.0291    0.0952    0.044

```