#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from sklearn.linear_model import LinearRegression
from numba import jit

@jit
def polynomial_this(x,y,n):
    X = np.c_[np.ones(len(x))]
    for i in range(1,n+1):
        X = np.c_[X,x**(i)]
        for j in range(i-1,0,-1):
            X = np.c_[X,(x**(j))*(y**(i-j))]  
        X = np.c_[X,y**(i)]
    return X
"""@jit
def OLS(x,y,z,n,deg):
    X = polynomial_this(x,y,deg)
    beta = np.linalg.inv(X.T.dot(X)).dot(X.T.dot(z))
    
    xnew = np.linspace(0,1,n)
    ynew = np.linspace(0,1,n)
    xnew , ynew = np.meshgrid(xnew,ynew)
    Xnew = polynomial_this(xnew.reshape(n**2,1),ynew.reshape(n**2,1),deg)
    znew = Xnew.dot(beta)
    plutt = znew.reshape((n,n))
    return xnew, ynew, plutt
"""
def OLS(x,y,z,n,deg):
    X = polynomial_this(x,y,deg)
    beta = np.linalg.inv(X.T.dot(X)).dot(X.T.dot(z))
    znew = X.dot(beta)
    plutt = znew.reshape((n,n))
    return plutt


def ridge(x,y,z,n,deg,lambd):
    X = polynomial_this(x,y,deg)
    beta = np.linalg.inv(X.T.dot(X)+lambd*np.identity(X.shape[1])).dot(X.T.dot(z))
    znew = X.dot(beta)
    plutt = znew.reshape((n,n))
    return plutt

def FRANK(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1+term2+term3+term4

#oppg 1
n = 500
deg = 5
x = np.sort((np.random.rand(n)))
y = np.sort((np.random.rand(n)))

x,y = np.meshgrid(x,y)

x1d = x.reshape((n**2,1))
y1d = y.reshape((n**2,1))


z = FRANK(x1d,y1d) #+5*np.random.randn(n*n,1)

z_plot = ridge(x1d,y1d,z,n,deg,0.1)
temp = z_plot.reshape((n**2,1))
true = FRANK(x,y).reshape((n**2,1))

MSE = sum((true-temp)**2)/(len(true))
R2  = 1-(np.sum((true - temp)**2)/np.sum((true-np.mean(true))**2))

from sklearn.metrics import r2_score

#print(max(z-temp))

print(r2_score(true, temp),"sklearn")
 

print(MSE,R2)

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
from matplotlib import cm
fig = plt.figure()
ax = fig.gca(projection="3d")
# Plot the surface.
surf = ax.plot_surface(x,y,z_plot, cmap=cm.coolwarm,
linewidth=0, antialiased=False)
# Customize the z axis.
ax.set_zlim(-0.10, 1.40)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))
# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()

