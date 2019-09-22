import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset 
dataset = pd.read_csv("data.csv")
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

#there is no need to divide into traning and testing data we have very less dataset

# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

#Fitting polynomial regression to the dataset 
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 =LinearRegression()
lin_reg_2.fit(X_poly,y)

#Visualising the linear regression results

plt.scatter(X,y,color='red')
plt.plot(X, lin_reg.predict(X), color='blue')
plt.title('Truth or bluff')
plt.xlabel('Position level')
plt.ylabel('salary')
plt.show()

#visualisation the polynomial regression model
#X_grid = np.arange(min(X), max(X), 0.1)
#X_grid = X_grid.reshape(len(X_grid), 1)


plt.scatter(X,y,color='red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color='blue')
plt.title('Truth or bluff')
plt.xlabel('Position level')
plt.ylabel('salary')
plt.show()


#predicting a new result with linear regression

lin_reg.predict(np.array([[10]]))

#predicting a new result with polynomial regression
lin_reg_2.predict(poly_reg.fit_transform(np.array([[10]])))

