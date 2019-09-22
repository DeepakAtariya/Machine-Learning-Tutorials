# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 20:16:53 2019

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
while(err>1.e-5):
    yh = (w1*X)+w0
#    DJ0 = (yh-Y)
    dw0 = (sum(yh-Y))/m
#    dw0 = (-2/m) * sum(Y - yh)
#    record_dw0_np.append(dw0)    
#    dw1 = (-2/m) * sum(X * Y - yh)
#    DJ1 = (yh-Y)*X
    dw1 = (sum((yh-Y)*X))/m
    w0 = w0 - alp*dw0
    w1 = w1 - alp*dw1
    itr = itr + 1
    print(w1,w0)
    j.append(0.5/m*sum((Y-w1*X-w0)**2))
    err = abs(j[itr]-j[itr-1])
    print("error "+str(err))
    plt.scatter(X,Y)
    plt.plot(X,yh,color='red') 
#    plt.show()
#    break
#real_yh = (w1*X)+w0
