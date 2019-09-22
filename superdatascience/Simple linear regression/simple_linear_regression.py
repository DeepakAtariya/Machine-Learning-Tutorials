#simple linear regression 

#importing the libraries 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Salary_Data.csv')

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values
 
#splitting the dataset into the training set and test set 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y ,test_size = 1/3, random_state = 0)


#feature scaling 

"""
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)
"""
#data preprocessing (import dataset -> convert into array -> split into training and test data -> feature scaling)

#fitting simple linear regression to the training set 
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

"""
Questions : 
    1. why regression analysis is important?
    2. simple linear regression formula - (y = B0+B1*X1) 
        how to find constant, regressor,
                                                
"""

#predicting the test set results

y_pred = regressor.predict(X_test)

# Visualising the traning set result
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('salary vs experience(Training set)')
plt.xlabel('experience')
plt.ylabel('salary')
plt.show()

# Visualising the test set result
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('salary vs experience(Tese set)')
plt.xlabel('experience')
plt.ylabel('salary')
plt.show()

