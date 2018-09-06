#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from random import random, seed
import numpy as np
import matplotlib.pyplot as plt

n = 100

x = 2*np.random.rand(n,1)
y = 4+3*x+np.random.randn(n,1)

xb = np.c_[np.ones((n,1)), x]

print(xb)

beta = np.linalg.inv(xb.T.dot(xb)).dot(xb.T).dot(y)
xnew = np.array([[0],[2]])
xbnew = np.c_[np.ones((2,1)), xnew]
ypredict = xbnew.dot(beta)

plt.plot(xnew, ypredict, "r-")
plt.plot(x, y ,'bo')
plt.axis([0,2.0,0, 15.0])
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.title(r'Linear Regression')
plt.show()