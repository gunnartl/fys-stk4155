#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np

def reg2(x,y):
    X = np.c_[np.ones(n),x,x**2]
    beta = np.linalg.inv(X.T.dot(X)).dot(X.T.dot(y))
    xnew = np.linspace(0,1,100)
    ynew = beta[0] + beta[1]*xnew + beta[2]*xnew*xnew
    return xnew, ynew


n= 100

plipp = np.random.rand(n)
plopp = 5*plipp*plipp+0.1*np.random.randn(n)

x,y = reg2(plipp,plopp)


import matplotlib.pyplot as plt
plt.scatter(plipp,plopp)
plt.plot(x,y,'r')
plt.show()