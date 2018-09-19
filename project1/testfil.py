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
@jit
def OLS(x,y,z,n,deg):
    X = polynomial_this(x,y,deg)
    beta = np.linalg.inv(X.T.dot(X)).dot(X.T.dot(z))
    
    xnew = np.linspace(0,1,n)
    ynew = np.linspace(0,1,n)
    xnew , ynew = np.meshgrid(xnew,ynew)
    Xnew = polynomial_this(xnew.reshape(n**2,1),ynew.reshape(n**2,1),deg)
    znew = Xnew.dot(beta)
    f = znew.reshape((n,n))
    return xnew, ynew, f

def FRANK(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1+term2+term3+term4

#oppg 1
n = 100
deg = 5
x = np.random.rand(n)
y = np.random.rand(n)

x,y = np.meshgrid(x,y)

x1d = x.reshape((n**2,1))
y1d = y.reshape((n**2,1))


z = FRANK(x1d,y1d)# +0.9*np.random.randn(n*n,1

x_plot,y_plot , z_plot = OLS(x1d,y1d,z,n,deg)

MSE = (sum((z-z_plot.reshape((n**2,1)))**2)/n**2)
R2  = 1-(sum(z - z_plot.reshape((n**2,1)))**2/sum(z-(sum(z)/n**2))) 

print(MSE,R2)


import matplotlib.pyplot as plt
from matplotlib import cm

# Plot the surface.
fig = plt.figure()
ax = fig.gca(projection="3d")

surf = ax.plot_surface(x_plot, y_plot, z_plot, cmap=cm.coolwarm,linewidth=0, antialiased=False)

plt.show()

