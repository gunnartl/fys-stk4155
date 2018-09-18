#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from sklearn.linear_model import LinearRegression

def reg2(x,y,z,n):
    X = np.c_[np.ones(n),
              x,y,
              x**2,x*y,y**2,
              x**3,x**2*y,x*y**2,y**3,
              x**4,x**3*y,x**2*y**2,x*y**3,y**4,
              x**5,x**4*y,x**3*y**2,x**2**y**3,x*y**4,y**5]
    print(X.shape)
    beta = np.linalg.inv(X.T.dot(X)).dot(X.T.dot(z))
    print(beta.shape)
    print(np.linalg.norm(beta[:,1]))
    
    xnew = np.linspace(0,1,n)
    ynew = np.linspace(0,1,n)
    xnew , ynew = np.meshgrid(xnew,ynew)
    znew = sum(beta[0,:]) + sum(beta[1,:])*xnew + sum(beta[2,:])*ynew + sum(beta[3,:])*xnew**2 + \
           sum(beta[4,:])*xnew*ynew + sum(beta[5,:])*ynew**2+sum(beta[6,:])*xnew**3+sum(beta[7,:])*xnew**2*ynew +\
           sum(beta[8,:])*xnew*ynew**2 + sum(beta[9,:])*ynew**3+sum(beta[10,:])*xnew**4+sum(beta[11,:])*xnew**3*ynew+\
           sum(beta[12,:])*xnew**2*ynew**2 + sum(beta[13,:])*xnew*ynew**3+sum(beta[14,:])*ynew**4+sum(beta[15,:])*xnew**5+\
           sum(beta[16,:])*xnew**4*ynew + sum(beta[17,:])*xnew**3*ynew**2+sum(beta[18,:])*xnew**2*ynew**3+sum(beta[19,:])*xnew*ynew**4+sum(beta[20,:])*ynew**5    
    #znew = 5
    return xnew, ynew, znew

def FRANK(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1+term2+term3+term4

#oppg 1
n = 1000
x = np.random.rand(n)
y = np.random.rand(n)
#x = np.linspace(0,1,n)
#y = np.linspace(0,1,n)
xs,ys = np.meshgrid(x,y)
#print(x.shape)
z = FRANK(xs,ys) #+0.000001*np.random.randn(n,n)
#print(y.shape)

x_plot,y_plot , z_plot = reg2(x,y,z,n)
#x_plot,y_plot = np.meshgrid(x_plot,y_plot)

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
# Plot the surface.
fig = plt.figure()
ax = fig.gca(projection="3d")
#ax.scatter(x,y,z,cmap=cm.coolwarm)
surf = ax.plot_surface(x_plot, y_plot, z_plot, cmap=cm.coolwarm,linewidth=0, antialiased=False)
# Customize the z axis.
#ax.set_zlim(-0.10, 1.40)
#ax.zaxis.set_major_locator(LinearLocator(10))
#ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))
# Add a color bar which maps values to colors.
#fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()


