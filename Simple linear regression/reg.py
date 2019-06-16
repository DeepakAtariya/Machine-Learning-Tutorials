import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

# import the dataset
dataset = pd.read_csv('data.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:, 3].values

# Splittin the dataset into training and testing 
from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# feature scaling 
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler
X_train