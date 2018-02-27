# Session2-Introduction to Classification
We start our first foray into Machine Learning - classifying some data using the SciKit Learn library.

We will use the [Iris Data Set](https://archive.ics.uci.edu/ml/datasets/iris). This is perhaps the best known database to be found in the pattern recognition literature. The data set contains 3 classes of 50 instances each, where each class refers to a type of iris plant. One class is linearly separable from the other 2; the latter are NOT linearly separable from each other. 

We will use this to train some popular supervised classification algorithms in Python and [SciKit-Learn](http://scikit-learn.org/stable/).

The first couple of Notebooks introduce you to the environment and look at the data set in more detail and shows how you can perform  some basic exploratory data analysis.

## Getting the Data
The data can be downloaded from the ICS archives but is also included with SciKit-Learn. To make life especially easy there is a helper module in the repo that has a function `get_iris_data()` that automatically loads the Iris data set for you - so we dont have to worry too much about how this is done today. The function also identifies which columns will be used as features (X) and labels (y).

### Test - Train Split
This function also splits the data into __Train__ data and __Test__ data. As the names suggest, we will use one set to train our model and then test the model on the other set. The y label values are also converted into integers to optimise performance. 

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

The function is as follows: 
*  `get_iris_data(test_size=0.3)`  -> Iris Data Set retrieval, split into train/test set (default 30%, can be changed by setting test_size parameter) and standardised. Returns the `X_train_std, y_train, X_test_std, y_test, X_combined_std, y_combined, X_train, X_test`

### Plotting the data 
The utility module also has a funcion that plots the data, highlighting the training and test data and plotting the model boundaries so we can visualise how each model looks against the data.

* `plot_decision_regions(X,y,classifier, test_idx=None, resolution=0.02)`  -> plots the data and the decision regions produced by a model


# TASKS
Some of you may already be familair with Python, Jupyter, Numpy, matplotlib and Pandas and data visualisation and so can move quickly through the first few notebooks. If you are less familiar we recommend working through these slowly and try to understand what is happening - you can always work through teh rest in your own time and ask questions when you are stuck!

Each notebook guides you through so you can go at your own pace - there are varies questions and challenges along the way!

1. Warm yourself up with Python, Jupyter, Numpy and matplotlib by investigating the Sigmoid Function.  
2. Familiarise youself with the Iris Dataset, including using Pandas to explore it and plot it.
3. Step through the Logistic Regression example - get the data,  prepare the data, fit the model, evaluate and predict


# EXTRA
1. What other ways could you visualise and explore the Iris dataset? Implement these and try different Pandas and matplotlib routines.
2. Rerun the Logistic regression classifier but change the random_state value. What is happening?
3. Rerun the regression classifier but change the C value. What is happening?
4. Open up the scikit_utilities.py file and have a deeper look at the functions. 

