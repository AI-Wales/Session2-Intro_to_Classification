# Session2-Introduction to Classification
We start our first foray into Machine Learning - classifying some data using the SciKit Learn library.

We will use the [Iris Data Set](https://archive.ics.uci.edu/ml/datasets/iris). This is perhaps the best known database to be found in the pattern recognition literature. The data set contains 3 classes of 50 instances each, where each class refers to a type of iris plant. One class is linearly separable from the other 2; the latter are NOT linearly separable from each other. 

We will use this to train some popular supervised classification algorithms in Python and [SciKit-Learn](http://scikit-learn.org/stable/).

The first Notebook looks at the data set in more detail and does some basic exploratory data analysis on it.

## Getting the Data
The data can be downloaded from the ICS archives but is also included with SciKit-Learn. There is a helper module in the repo that has a function `get_iris_data()` that automatically loads the Iris data set for you - so we dont have to worry too much about how this is done today. The function also identifies which columns will be used as features (X) and labels (y).

### Test - Train Split
This function also splits the data into __Train__ data and __Test__ data. As the names suggest, we will use one set to train our model and then test the model on the other set.

Data splitting is performed using the [test_train_split](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) utility in SciKit-Learn, defaulting with a test size of 30%.

```Python
 X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
```

### Data Standardisation
The function also standardises the train data features (X) - centering them on 0 with a standard deviation of 1. We could normalise the data (so that it fits between 0 and 1) but normalisation can sometimes loose information about outliers.
We standardise the data as this makes many algorithms far more efficient. 
 
The [StandardScaler](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) is used for standardisation - the fit function estimates the sample mean and std deviation for each feature dimension in the training data, and these are used to transform the training and test data sets. 

```Python
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
```

*  `get_iris_data(test_size=0.3)`  -> Iris Data Set retrieval, split into train/test set (default 30%, can be changed by setting test_size parameter) and standardised. Returns the `X_train_std, y_train, X_test_std, y_test, X_combined_std, y_combined, X_train, X_test`

### Plotting the data 
* `plot_decision_regions(X,y,classifier, test_idx=None, resolution=0.02)`  -> plots the data and the decision regions produced by a model


# TASKS
1. Work through the data notebook and get to know the Iris Dataset
2. Have a look at the Sigmoid Function - this is a typical function used to help make binary decisions.
3. Step through the Logistic Regression example - follow each step using the system diagram below:

4. Step through the K-NN example - follow each step using the system diagram below:
5. Compare the accuracy of the two models - is one better than the other? Would you be happy with these models and predictions? Also look at the visual plot of the model. Are they similar? Can you - by eye - come up with a better set of regions to catorgise the data into?



# EXTRA
1. What other ways could you visualise and explore the Iris dataset?
2. What affect does the NN paramter have on the model? Run run teh code with different XX and see what happens. How could you optimise this?

