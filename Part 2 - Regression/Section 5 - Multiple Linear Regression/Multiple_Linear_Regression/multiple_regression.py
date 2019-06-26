# -*- coding: utf-8 -*-

# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('startup.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

#encoding categorical data
# encoding the independant variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,3] = labelencoder_X.fit_transform(X[:,3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

#Avoiding the dummy variable trap
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
#distributing the data into traning and the test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Fitting Multiple Linear Regression the training set
from sklearn.linear_model import LinearRegression 
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the test set results
# In the case we used all the independent variables
y_pred = regressor.predict(X_test)

"""
Building the optimal model using backward elimination
backward elimation method eliminates from largers data into smallers ones till Significance level satisfy 

"""

import statsmodels.formula.api as sm 
X = np.append(arr=np.ones((50,1)).astype(int), values = X, axis = 1)

# backward elimination method
X_opt = X[:, [0,1,2,3,4,5]]
#OLS = Ordinary Least Square
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
summary = regressor_OLS.summary()

X_opt = X[:, [0,1,2,4,5]]
#OLS = Ordinary Least Square
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
summary = regressor_OLS.summary()


X_opt = X[:, [0,1,4,5]]
#OLS = Ordinary Least Square
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
summary = regressor_OLS.summary()


X_opt = X[:, [0,1,4]]
#OLS = Ordinary Least Square
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
summary = regressor_OLS.summary()



