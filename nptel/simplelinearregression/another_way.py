# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 14:45:26 2019

@author: Deepak
"""

import numpy as np
import pandas as pd
#import random as rd
import matplotlib.pyplot as plt
#from sklearn.datasets.samples_generator import make_blobs
#from sklearn.model_selection import train_test_split

data = pd.read_csv('data.csv')


#x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 1/3, random_state = 0)


X = data.iloc[:,0]
Y = data.iloc[:,1]


#plt.scatter(x,y)
#plt.show()
#

record_dw0_np = list()
record_dw0_td = list()



j = list()
w1 = 3
w0 = 4
alp = 0.0001
m = len(X)
j.append(0.5/m*sum((Y-w1*X-w0)**2))
err = 1
itr = 0

# Building the model

#L = 0.0001  # The learning Rate
epochs = 5  # The number of iterations to perform gradient descent

n = float(len(X)) # Number of elements in X

# Performing Gradient Descent 
for i in range(epochs): 
    Y_pred = w1*X + w0  # The current predicted value of Y
    D_m = (-2/n) * sum(X * (Y - Y_pred))  # Derivative wrt m
    
    D_c = (-2/n) * sum(Y - Y_pred)  # Derivative wrt c
    record_dw0_td.append(D_c)
    w1 = w1 - alp * D_m  # Update m
    w0 = w0 - alp * D_c  # Update c
#    plt.show()
    print (w1, w0)

#real_yh = (w1*X)+w0
