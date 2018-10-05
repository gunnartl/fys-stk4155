#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
from sklearn.model_selection import train_test_split
import reg_class as Reg2
from reg_class import FRANK, KfoldCrossVal

n = 100
Numbdeg = 15
Numbfolds = 100

deg = Numbdeg
x = np.sort((np.random.rand(n)))
y = np.sort((np.random.rand(n)))

x, y = np.meshgrid(x,y)

x1d = x.reshape((n**2, 1))
y1d = y.reshape((n**2, 1))

error = 0.1*np.random.randn(n**2, 1)

z = FRANK(x1d,y1d) #+ error

X = Reg2.polynomial_this(x1d, y1d, deg)



X, X_holdout, z, z_holdout = train_test_split(X, z, test_size = 0.2)
x_holdout = X_holdout[:,1]
y_holdout = X_holdout[:,2]

X_Split, z_Split = KfoldCrossVal(X, z, Numbfolds)

z_mean =np.zeros((Numbfolds,Numbdeg,len(X_holdout)))

for i in range(Numbfolds):
    XTrainSets = np.delete(X_Split, i, 0)
    XTestSets = X_Split[i]
    zTrainSets = np.delete(z_Split, i, 0)
    zTestSets = z_Split[i]
    XTrainSets = np.vstack(XTrainSets)
    zTrainSets = np.vstack(zTrainSets)
   

    for k in range(Numbdeg):
        #Xiterate = Reg2.polynomial_this(XTrainSets[:,1],XTrainSets[:,2], k)
        cumsum = int((k+2)*(k+1)/2)
        Xiterate = XTrainSets[:,:cumsum]
        #print
        #Xiterate = XTrainSets[:]
        #print(Xiterate.shape,int((k+2)*(k+1)/2),k)
        Frankiterate = Reg2.regression(Xiterate, zTrainSets)
        betas = Frankiterate.ridge(0.001)
        X_holdout_k = X_holdout[:,:cumsum]#Reg2.polynomial_this(x_holdout,y_holdout,k)
        zPred = np.dot(X_holdout_k, betas)

        z_mean[i,k,:] = zPred.squeeze()


 
z_mean_mean = np.mean(z_mean,axis=0)
error = np.zeros(Numbdeg-1)
bias  = np.zeros_like(error)
variance = np.zeros_like(error)

print(z_mean.shape)

z_holdout = z_holdout.squeeze()
#%%
error    = np.mean(np.mean((z_holdout-z_mean)**2, axis = 0),axis=1)
bias     = np.mean((z_holdout-np.mean(z_mean,axis=0))**2,axis=1)
variance = np.mean(np.var(z_mean,axis=0),axis=1)
print(np.mean((z_holdout-z_mean)**2,axis=0).shape)
import matplotlib.pyplot as plt
plt.plot(error,"--")
plt.plot(bias)
plt.plot(variance)

plt.legend(["error","bias","variance"])
plt.semilogy()
#plt.semilogx()
plt.show()
