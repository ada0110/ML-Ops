# ML-Ops

> To avoid folder name confusion, I have put code in `src` folder.


To run the Classification example follow below instructions:

1. Create a new conda environment  
```conda create --name new_env_name python=3.6```

2. Switch to this environment  
```conda activate new_env_name```

3. To install requirements, use conda. This will ensure that dependencies are resolved correctly.  
```conda install --file requirements.txt```

**Initial Output**

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

**Results on Rescaling and Resizing Images**

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
<!-- 
**Changing hyperparameter and observing results for **
```
gamma     accuracy   precision    recall        f1
0.0005    0.9556      0.9565      0.9555      0.9554
0.001     0.9704      0.9709      0.9704      0.9703
0.005     0.8889      0.9432      0.8885      0.9022
0.01      0.6963      0.9241      0.6971      0.7544
0.05      0.1019      0.1099      0.1038      0.0252
0.1       0.0981      0.0098      0.1         0.0179
``` -->

**Results on changing gamma hyperparameter and observing results for 70:15:15 split**

> The scores below are in the order: [accuracy, precision, recall, f1]

```
shape of data: (1797, 8, 8)
shape of single image: (8, 8)


gamma: 5e-06
        train scores:   [0.7486, 0.8064, 0.7449, 0.6943]
        test scores:    [0.7037, 0.6482, 0.6929, 0.6485]
        val scores:     [0.7037, 0.6598, 0.7093, 0.6514]

gamma: 0.0001
        train scores:   [0.9857, 0.9859, 0.9856, 0.9856]
        test scores:    [0.9556, 0.9639, 0.955, 0.9546]
        val scores:     [0.9, 0.9108, 0.8998, 0.8977]

gamma: 0.0005
        train scores:   [0.9976, 0.9976, 0.9976, 0.9976]
        test scores:    [0.9852, 0.9858, 0.9849, 0.985]
        val scores:     [0.9259, 0.9314, 0.9259, 0.925]

gamma: 0.001
        train scores:   [0.9992, 0.9992, 0.9992, 0.9992]
        test scores:    [0.9926, 0.993, 0.9926, 0.9926]
        val scores:     [0.9481, 0.9515, 0.9479, 0.9477]

gamma: 0.005
        train scores:   [1.0, 1.0, 1.0, 1.0]
        test scores:    [0.9333, 0.9581, 0.9335, 0.9371]
        val scores:     [0.8444, 0.9306, 0.8445, 0.8639]

gamma: 0.01
        train scores:   [1.0, 1.0, 1.0, 1.0]
        test scores:    [0.7556, 0.9275, 0.7588, 0.7955]
        val scores:     [0.637, 0.9216, 0.6353, 0.7004]

gamma: 0.05
        train scores:   [1.0, 1.0, 1.0, 1.0]
        test scores:    [0.1037, 0.11, 0.1037, 0.0254]
        val scores:     [0.1, 0.1097, 0.1038, 0.025]

gamma: 0.1
        train scores:   [1.0, 1.0, 1.0, 1.0]
        test scores:    [0.1, 0.01, 0.1, 0.0182]
        val scores:     [0.0963, 0.0096, 0.1, 0.0176]


best validation f1 score is 0.9477 for optimal gamma 0.001

Optimal gamma: 0.001
        train scores:   [0.9992, 0.9992, 0.9992, 0.9992]
        test scores:    [0.9926, 0.993, 0.9926, 0.9926]
        val scores:     [0.9481, 0.9515, 0.9479, 0.9477]
```

**Observations**
1. After resizing the images to lower dimensions, the numbers for accuracy, precision, recall and f1 are decreasing. This is due to loss in information.
2. With respect to train-test split, for `8*8`, `7*7` and `6*6`, the numbers are better for `70:30` split. In general, as we decrease training samples, numbers go down.
3. As we increase the gamma values from `5e-06` to `0.1`, the scores increase and then decrease. For gamma = `0.001`, we get the best f1 score. We found the optimal gamma using `70:15:15` train:test:validation split.



<br/>


**Saving and Predicting from Best model**

- For `gamma = 0.001` we get better scores and for that gamma we save the model and make predictions on the saved model. 

```
shape of data: (1797, 8, 8)
shape of single image: (8, 8)


data: (1797, 64)

(x, y) shape: (1797, 64) (1797,)
(x_train, y_train) shape: (1257, 64) (1257,)
(x_test, y_test) shape: (270, 64) (270,)
(x_val, y_val) shape: (270, 64) (270,)


gamma: 5e-06
        train scores:   [0.7486, 0.8064, 0.7449, 0.6943]
        val scores:     [0.7037, 0.6598, 0.7093, 0.6514]


gamma: 0.0001
        train scores:   [0.9857, 0.9859, 0.9856, 0.9856]
        val scores:     [0.9, 0.9108, 0.8998, 0.8977]


gamma: 0.0005
        train scores:   [0.9976, 0.9976, 0.9976, 0.9976]
        val scores:     [0.9259, 0.9314, 0.9259, 0.925]


gamma: 0.001
        train scores:   [0.9992, 0.9992, 0.9992, 0.9992]
        val scores:     [0.9481, 0.9515, 0.9479, 0.9477]


gamma: 0.005
        train scores:   [1.0, 1.0, 1.0, 1.0]
        val scores:     [0.8444, 0.9306, 0.8445, 0.8639]


gamma: 0.01
        train scores:   [1.0, 1.0, 1.0, 1.0]
        val scores:     [0.637, 0.9216, 0.6353, 0.7004]

>> skipping for gamma: 0.05 as 0.025 is less than 0.11
>> skipping for gamma: 0.1 as 0.0176 is less than 0.11


Saving the best model...


best validation f1 score is 0.9477 for optimal gamma 0.001
        test scores:    [0.9926, 0.993, 0.9926, 0.9926]


loading the model:
SVC(gamma=0.001)

predicting from loaded model:
        test scores:    [0.9926, 0.993, 0.9926, 0.9926] 
```
<br/>

**Pytest Experiment Output**  

```
(py3.6) ada@LAPTOP-U0O9E34L:~/codes/ML-Ops_Scikit/tests$ pytest
=============================== test session starts ===============================
platform linux -- Python 3.6.9, pytest-6.2.5, py-1.10.0, pluggy-1.0.0
rootdir: /home/ada/codes/ML-Ops_Scikit/tests
plugins: anyio-3.3.0
collected 5 items

test_exampes.py ...                                                         [ 60%]
test_model.py ..                                                            [100%] 

================================ 5 passed in 1.09s ================================
(py3.6) ada@LAPTOP-U0O9E34L:~/codes/ML-Ops_Scikit/tests$ 
```