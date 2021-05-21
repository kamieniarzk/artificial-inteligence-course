## Dataset 
The dataset used for training the classifiers was based on quality of wines, it is available [here](https://www.kaggle.com/uciml/red-wine-quality-cortez-et-al-2009). The input consists of 11 numerical variables, including:
* `fixed acidity`
* `volatile acidity`
* `citric acid`
* `residual sugar`
* `chlorides`
* `free sulfur dioxide`
* `total sulfur dioxide`
* `density`
* `pH`
* `sulphates`
* `alcohol`

The output variable `quality` is an integer between 0 and 10. 

There are 2 classifiers implemented in our solution: Support Vector Machine and Random Forest. The whole dataset consists of 1599 samples. 

We've devided the whole dataset into 3 subsets and randomly distributed 1599 samples in the following proportions:
* 10% went into `test` set,
* 10% went into `validation` set,
* 80% went into `train` set.

## Evaluation metrics
To assess our classifiers, We've used several metrics, including:

## Support vector machine
For implementing the classifiers, we used a library called [scikit-learn](https://scikit-learn.org/stable/). The hyperparameters that were taken into account while assessing the classifier's performance were:
* kernel, 
* dropout rate,
* coefficient, degree (for polynomial kernel),
* gamma.

In order to avoid overfitting on the test set, for tuning of the parameters We've used the validation set in the process of cross-validation. At first, We've picked three kernel types, including polynomial, Gaussian RBF and linear. For cross-validation, We've used the function `cross_val_score` which takes number of dataset divisions as an argument and outputs the evaluation for each subest of the division. 

* `Rbf CV accuracy: 0.48 % with the standard deviation: 0.04`  
* `Polynomial CV accuracy: 0.51 % with the standard deviation: 0.02`  
* `Linear CV accuracy: 0.54 % with the standard deviation: 0.08`


For all of the three kernels, We've chosen such hyperparameters that yielded best results, which was:
* For RBF - (gamma = 'auto', C = 0.8)
* for polynomial - (C = 0.8, degree = 3, coefficient = 2),
* for linear - (C = 0.8).

For given set of hyperparameters, the results are:  
```python
RBF Kernel:
Accuracy:  60.62 %
Recall: 60.62 %
F1: 57.70 %
Confusion Matrix:
[[ 0  4  3  0  0]
 [ 0 41 22  0  0]
 [ 0 25 42  0  0]
 [ 0  1 18  0  0]
 [ 0  1  3  0  0]]

Polynomial Kernel:
Accuracy: 53.75 %
Recall: 53.75 %
F1: 49.94 %
Confusion Matrix:
[[ 0  0  2  0]
 [ 0 44 28  0]
 [ 0 17 41  0]
 [ 0  1 26  1]]

Linear Kernel:
Accuracy: 54.37 %
Recall: 54.37 %
F1: 49.23 %
Confusion Matrix:
[[ 0  1  1  0]
 [ 0 54 18  0]
 [ 0 25 33  0]
 [ 0  1 27  0]]

```
Comparing those metrics, We can observe, that the best results for the SVM classifier were obtained using the **RBF kernel**.

## Random forest
Another classifier that we've used in our excerise was Random forest classifier. Similarly like in the previous one, the implementation is from the scikit-learn library. In case of this classifier, the only hypermarameter which has been considered was `n_estimators` which represents the number of trees in the forest. After the cross-validation process, the optimal value of `n_estimators` obtained was 100. Increasing the number over 100 did not have any effect on the validation and final evaluation.

* `Random forest CV accuracy: 0.63 % with the standard deviation: 0.07`

The metrics obtained for such value of the hyperparameter were:

```python
Random forest:
Accuracy: 70.62 %
Recall: 70.62 %
F1: 68.78 %
Confusion Matrix:
[[ 0  5  2  0  0]
 [ 1 53  8  1  0]
 [ 0 18 47  2  0]
 [ 0  1  6 12  0]
 [ 0  0  1  2  1]]
```

## k-nearest neighbour classifier
The third classifier, that we've implemented was k-nearest neighbour classifier, also using the scikit-learn library. The only hyperparameter that considerably changed the obtained results was `n_neighbours` which determined the number of neighbours taken into consideration when predicting the class of the test sample. After cross-validation, the best results were obtained for `n_neighbours = 5`.

* `KNN CV accuracy: 0.44 % with a standard deviation of: 0.02`

The evaluation for the setup above was: 

```python
K-Nearest neighbors:
Accuracy: 54.37 %
Recall: 54.37 %
F1: 51.51 %
Confusion Matrix:
[[ 0  5  2  0  0]
 [ 0 42 20  1  0]
 [ 1 23 41  2  0]
 [ 0  2 13  4  0]
 [ 0  2  1  1  0]]
```

## Final remarks
Although for each type of classifier, we've managed to extract such setups (hyperparameters) that were performing best in most situations, it is important to note, that the overall results were fluctuating over each iteration. This indicates, that the performance of the classifiers was highly dependent on the division of the dataset (as it was randomly distributed in each iteration). Comparing all the used classifiers, it was clear, that the best performance overall was provided by the Random forest classifier.