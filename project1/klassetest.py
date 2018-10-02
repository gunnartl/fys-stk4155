#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from sklearn.linear_model import LinearRegression
from numba import jit
from reg_class import regression,polynomial_this
from sklearn import linear_model


"""
@jit
def polynomial_this(x,y,n):
    X = np.c_[np.ones(len(x))]
    for i in range(1,n+1):
        X = np.c_[X,x**(i)]
        for j in range(i-1,0,-1):
            X = np.c_[X,(x**(j))*(y**(i-j))]  
        X = np.c_[X,y**(i)]
    return X

def OLS(x,y,z,n,deg):
    X = polynomial_this(x,y,deg)
    beta = np.linalg.inv(X.T.dot(X)).dot(X.T.dot(z))
    znew = X.dot(beta)
    plutt = znew.reshape((n,n))
    return plutt

class regression:
    def __init__(self,x,y,z,n,deg):
        self.x = x
        self.y = y
        self.z = z
        self.n = n
        self.deg = deg
        self.X = polynomial_this(x,y,deg)
        
    @jit    
    def ridge(self,lambd):#,x,y,z,n,deg,lambd):
        X = self.X
        beta = np.linalg.inv(X.T.dot(X)+lambd*np.identity(X.shape[1])).dot(X.T.dot(self.z))
        self.znew = X.dot(beta)
        #plutt = znew.reshape((self.n,self.n))
        self.beta = beta
        return beta#plutt
    
    @jit
    def OLS(self):
        X = self.X
        beta = np.linalg.inv(X.T.dot(X)).dot(X.T.dot(self.z))
        self.znew = X.dot(beta)
        #plutt = znew.reshape((n,n))
        return beta
    
    @jit
    def lasso(self, alphain):
        #from sklearn import linear_model
        #from sklearn.linear_model import LinearRegression
        lasso = linear_model.Lasso(alpha = alphain,fit_intercept = False)
        lasso.fit(self.X, self.z)
        print("HER ER JEG")
        beta = lasso.coef_
        self.znew = self.X.dot(beta)
        return beta         
    def variance(self):
        e =7
        
    def plot(self):
        plutt = self.znew.reshape((self.n,self.n))
        return plutt
    
    def MSE(self):
        MSE = sum((self.z-self.znew)**2)/(len(self.z))
        return MSE
    
    def R2(self):
        self.R2 = 1-(np.sum((self.z - self.znew)**2)/np.sum((self.z-np.mean(self.z))**2))
        return self.R2

"""
def FRANK(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1+term2+term3+term4
"""
from imageio import imread
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
# Load the terrain
terrainfull = np.array(imread("SRTM_data_Norway_1.tif"))
terrain1 = terrainfull[1000:2000,800:1800]
tx = np.linspace(0,1,terrain1.shape[0])
ty = np.linspace(0,1,terrain1.shape[1])

tx,ty = np.meshgrid(tx,ty)

tx1d = tx.ravel()
ty1d = ty.ravel()
terrain11d=terrain1.ravel()

print(terrain1.shape,tx.shape,ty.shape)
deg = 20
terreng = regression(tx1d,ty1d,terrain11d,5,deg)
terreng_beta = terreng.ridge(0.001)
print(terreng_beta)
TX = polynomial_this(tx1d,ty1d,deg)
terengplutt = TX.dot(terreng_beta)
terengplutt = terengplutt.reshape((terrain1.shape[0],terrain1.shape[1]))

print(terreng.R2(),"THOMMASERBEST")
# Show the terrain

from matplotlib.ticker import LinearLocator, FormatStrFormatter
fig = plt.figure()
ax = fig.gca(projection="3d")
# Plot the surface.
surf = ax.plot_surface(-tx,-ty,terrain1, cmap=cm.coolwarm,
linewidth=0, antialiased=False)
# Customize the z axis.
#ax.set_zlim(-0.10, 1.40)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))
# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.hold("on")
#plt.show()

plt.figure()
plt.title("Terrain_over_Norway_1")
plt.imshow(terrain1, cmap="gray")
plt.xlabel("X")
plt.ylabel("Y")
plt.hold("on")
#plt.show()
#oppg 1"""
n = 100
deg = 5
x = np.sort((np.random.rand(n)))
y = np.sort((np.random.rand(n)))

x,y = np.meshgrid(x,y)

x1d = x.reshape((n**2,1))
y1d = y.reshape((n**2,1))


z = FRANK(x1d,y1d) #+0.5*np.random.randn(n*n,1)


frankreg = regression(x1d,y1d,z,n,deg)
betaols = frankreg.OLS()
OLSMSE = frankreg.MSE()
print(OLSMSE)
OLSBIAS = frankreg.bias()
OLSvariance = np.mean(frankreg.variance())
term = frankreg.term()
print(OLSBIAS, "bias")
print(OLSvariance, "varians")
print(OLSBIAS+OLSvariance+term)

"""
ridge = regression(x1d,y1d,z,n,deg)
ridge_beta = ridge.ridge(0.01)
z_plot = ridge.plot()
temp = z_plot.reshape((n**2,1))
true = FRANK(x,y).reshape((n**2,1))"""

#MSE = sum((true-temp)**2)/(len(true))
#R2  = 1-(np.sum((true - temp)**2)/np.sum((true-np.mean(true))**2))

from sklearn.metrics import r2_score

#print(max(z-temp))

print(r2_score(z, frankreg.plot().ravel()),"sklearn")
"""
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
from matplotlib import cm
fig = plt.figure()
ax = fig.gca(projection="3d")
# Plot the surface.
surf = ax.plot_surface(-tx,-ty,terengplutt, cmap=cm.coolwarm,
linewidth=0, antialiased=False)
# Customize the z axis.
#ax.set_zlim(-0.10, 1.40)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))
# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()"""