#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
from sklearn.model_selection import train_test_split
import reg_class as Reg
from reg_class import FRANK, KfoldCrossVal

n = 50
Numbdeg = 8
Numbfolds = 5
lambd = 0.00001

"""
Optimal OLS   : for 50x50 numbdeg = 8, R2 ~ 0.89, MSE ~ 0.01 ::: etter 8 klikker den

for Ridge og Lasso er det egentlig bare å ha en ganske lav lambda ~ 1e-5 eller lavere 
for å få et ca like bra resultat som OLS. høyere grad på polynom gir ikke særlig bedre resultat

"""

deg = Numbdeg
x = np.sort((np.random.rand(n)))
y = np.sort((np.random.rand(n)))

x, y = np.meshgrid(x,y)

x1d = x.reshape((n**2, 1))
y1d = y.reshape((n**2, 1))

error = 0.1*np.random.randn(n**2, 1)

z = FRANK(x1d,y1d) + error

X = Reg.polynomial_this(x1d, y1d, deg)



X, X_holdout, z, z_holdout = train_test_split(X, z, test_size = 0.2)
x_holdout = X_holdout[:,1]
y_holdout = X_holdout[:,2]

X_Split, z_Split = KfoldCrossVal(X, z, Numbfolds)

z_mean =np.zeros((Numbfolds,Numbdeg,len(X_holdout)))

R2  = np.zeros(Numbfolds)
MSE = np.zeros(Numbfolds)

    
for i in range(Numbfolds):
        XTrainSets = np.delete(X_Split, i, 0)
        XTestSets = X_Split[i]
        zTrainSets = np.delete(z_Split, i, 0)
        zTestSets = z_Split[i]
        XTrainSets = np.vstack(XTrainSets)
        zTrainSets = np.vstack(zTrainSets)
        
        FRANK = Reg.regression(XTrainSets,zTrainSets)
        betas = FRANK.lasso(lambd)                     # endre metode her ved å skrive OLS, ridge eller lasso (ols tar ikke argument) 
        
        
        TEST   = X_holdout.dot(betas) 
        R2[i]  += Reg.R2(z_holdout,TEST)
        MSE[i] += Reg.MSE(z_holdout,TEST)
    
R2var  = np.var(R2)
MSEvar = np.var(MSE)

print(R2,MSE)



