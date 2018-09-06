#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from sklearn.linear_model import LinearRegression

def reg2(x,y):
    X = np.c_[np.ones(n),x,x**2]
    beta = np.linalg.inv(X.T.dot(X)).dot(X.T.dot(y))
    xnew = np.linspace(0,1,100)
    ynew = beta[0] + beta[1]*xnew + beta[2]*xnew*xnew
    return xnew, ynew


#oppg 1
n= 100
x = np.random.rand(n)
y = 5*x*x+0.1*np.random.randn(n)

x_plot,y_plot = reg2(x,y)


import matplotlib.pyplot as plt
plt.scatter(x,y)
plt.plot(x_plot,y_plot,'r')
#plt.show()
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

X= x[:,np.newaxis]
X_plot= x_plot[:,np.newaxis]
model = make_pipeline(PolynomialFeatures(2), Ridge())
model.fit(X,y)
y_plot2 = model.predict(X_plot)
plt.plot(x_plot,y_plot2)
plt.show()