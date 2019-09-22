import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Data.csv')

X = dataset.iloc[:,: -1].values
y = dataset.iloc[:,3].values

#print(y)

# now we are going towards fixing the missing data from dataset

# missing values can be found using mean, median or other methods we have to find mean of particular row

# we use sklearn.preprocessing.Imputer to fix missing values 

# it replace missing values with average of column
from sklearn.preprocessing import Imputer 
imputer = Imputer(missing_values='NaN', strategy = 'mean', axis=0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])
#print(x)



#Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
#labeling the first column 
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
labelencoder_y = LabelEncoder()
y = labelencoder_X.fit_transform(y)

#Splitting the dataset into the Training set and Test set 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Feature Scaling 
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train_1 = sc_X.fit_transform(X_train)
X_test_1 = sc_X.transform(X_test)