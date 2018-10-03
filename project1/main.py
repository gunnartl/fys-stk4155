#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
from sklearn.linear_model import LinearRegression
from numba import jit
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn import linear_model
import Reg2
from Reg2 import MSE, variance, R2, bias
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
from matplotlib import cm

np.random.seed(2)

def FRANK(x, y):
    term1 = 0.75*np.exp(-(0.25*(9*x - 2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x + 1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

n = 100
deg = 5
x = np.sort((np.random.rand(n)))
y = np.sort((np.random.rand(n)))

x, y = np.meshgrid(x,y)

x1d = x.reshape((n**2, 1))
y1d = y.reshape((n**2, 1))

error = 0.01*np.random.randn(n**2, 1)

z = FRANK(x1d,y1d) + error

X = Reg2.polynomial_this(x1d, y1d, deg)

def KfoldCrossVal(dataset, dataset2, Numbfold):
    indices = np.arange(len(dataset[:, 0]))
    random_indices = np.random.choice(indices, size = len(dataset[:, 0]), replace = False)
    interval = int(len(dataset[:, 0])/Numbfold)
    datasetsplit = []
    dataset2split = []
    for k in range(Numbfold):
        datasetsplit.append(dataset[random_indices[interval*k : interval*(k + 1)]]) 
        dataset2split.append(dataset2[random_indices[interval*k : interval*(k + 1)]])

    return np.asarray(datasetsplit), np.asarray(dataset2split) 



lambdas = [10e-4 , 10e-3, 10e-2, 10e-1, 10e0, 10e1, 10e2, 10e3, 10e4]

def R2lambda(dataset1, dataset2, lambdas):
    Franklamb = Reg2.regression(dataset1, dataset2)
    R2ridge = []
    R2lasso = []
    R2OLS = []
    for lambd in lambdas:
        betasR = Franklamb.ridge(lambd)
        zRidge = np.dot(X, betasR)
        R2ridge.append(r2_score(z, zRidge))

        betasL = Franklamb.lasso(lambd)
        zLasso = np.dot(X, betasL)
        R2lasso.append(r2_score(z, zLasso))

        betasOLS = Franklamb.OLS()
        zOLS = np.dot(X, betasOLS)
        R2OLS.append(r2_score(z, zOLS))
    return R2OLS, R2ridge, R2lasso

#R2OLSTrain, R2ridgeTrain, R2lassoTrain = R2lambda(np.vstack(X_Split[0 :,:]), np.vstack(z_Split[0 :, :]), lambdas)
#R2OLSTest, R2ridgeTest, R2lassoTest = R2lambda(X_Split[0 , :], z_Split[0 , :], lambdas)

###plt.plot(np.log10(lambdas),R2ridgeTest, '-' ,  label = 'Ridge Test', color = 'red')
##plt.plot(np.log10(lambdas),R2OLSTest, '-' ,  label = 'OLS Test', color = 'blue')
#plt.plot(np.log10(lambdas), R2lassoTest, '-' ,  label = 'Lasso Test', color = 'green')
#plt.plot(np.log10(lambdas),R2ridgeTrain, '--' ,  label = 'Ridge Train', color = 'red')
#plt.plot(np.log10(lambdas),R2OLSTrain, '--' ,  label = 'OLS Train', color = 'blue')
#plt.plot(np.log10(lambdas), R2lassoTrain, '--' ,  label = 'Lasso Train', color = 'green')
#plt.xlabel(r'$log_{10}(\lambda)$', fontsize = 16)
#plt.ylabel(r'R2', fontsize = 16)
#plt.legend()  

beta_meanOLS = np.zeros((len(X[0, :]), 1))
beta_meanRidge = np.zeros((len(X[0, :]), 1))
beta_meanLasso = np.zeros((len(X[0, :]), 1))

Numbdeg = 20
Variterate = np.zeros(Numbdeg)
Biasiterate = np.zeros(Numbdeg)
MSEiterate = np.zeros(Numbdeg)

X, X_holdout, z, z_holdout = train_test_split(X, z, test_size = 0.2, random_state = 40)
x_holdout = X_holdout[:,1]
y_holdout = X_holdout[:,2]

Numbfolds = 10
X_Split, z_Split = KfoldCrossVal(X, z, Numbfolds)

z_mean =np.zeros((Numbfolds,Numbdeg,2000))
print(z_mean[0,0].shape,"DENNNNNE")
for i in range(Numbfolds):
    XTrainSets = np.delete(X_Split, i, 0)
    XTestSets = X_Split[i]
    zTrainSets = np.delete(z_Split, i, 0)
    zTestSets = z_Split[i]
    XTrainSets = np.vstack(XTrainSets)
    zTrainSets = np.vstack(zTrainSets)
    #Frank = Reg2.regression(XTrainSets, zTrainSets)

    #betaTrainOLS = Frank.OLS()
    #beta_meanOLS += betaTrainOLS
    
    #betaTrainRidge = Frank.ridge(0.1)
    #beta_meanRidge += betaTrainRidge

    #betaTrainLasso = Frank.lasso(0.0001)
    #beta_meanLasso += betaTrainLasso 
   

    for k in range(1, Numbdeg):
        Xiterate = Reg2.polynomial_this(XTrainSets[:,1],XTrainSets[:,2], k)

        #z = FRANK(x1d, y1d)
        #X_train, X_test, z_train, z_test = train_test_split(Xiterate, z, test_size = 0.2, random_state = 40)
        
        Frankiterate = Reg2.regression(Xiterate, zTrainSets)
        betas = Frankiterate.OLS()
        X_holdout_k = Reg2.polynomial_this(x_holdout,y_holdout,k)
        zPred = np.dot(X_holdout_k, betas)
        #print(zPred.shape, z_mean.shape)
        #exit()
        z_mean[i,k,:] = list(zPred)
        #MSEiterate[k] = MSE(z_test, zPred) 
        #Biasiterate[k] = bias(z_test, zPred)
        #Variterate[k] = variance(zPred)
       		#print(MSEiterate[k], Variterate[k], Biasiterate[k], k)

z_mean = np.mean(z_mean,axis=0)
error = np.zeros(len(range(1,Numbdeg)))
bias  = np.zeros_like(bias)
variance = np.zeros_like(bias)
for i in range(1,Numbdeg):      		
	error[i]    = np.mean( np.mean((z_holdout - z_mean[:,i,:])**2, axis=0, keepdims=True))
	bias[i]     = np.mean( (y_holdout - np.mean(z_mean, axis=0, keepdims=True))**2 )
	variance[i] = np.mean( np.var(z_mean, axis=0, keepdims=True) )
"""
plt.plot(np.linspace(1, Numbdeg, Numbdeg), Variterate, '*', label = 'Var')
plt.plot(np.linspace(1, Numbdeg, Numbdeg), Biasiterate, '--', label = 'Bias^2')
plt.plot(np.linspace(1, Numbdeg, Numbdeg), MSEiterate, 'o', label = 'MSE')
plt.legend()
plt.show()
exit()

#OLS evaluated on Franke function
beta_meanOLS /= Numbfolds
zOLSmean = np.dot(X, beta_meanOLS)
R2ScoreOLS = r2_score(z, zOLSmean)
MSEOLS = Frank.MSE()[1]
BiasOLS = Frank.Bias()[1]
VarOLS = Frank.Variance()[1]
#print(MSEOLS, BiasOLS, VarOLS)

#Ridge evaluated on Franke function
beta_meanRidge /= Numbfolds
zRidgemean = np.dot(X, beta_meanRidge)
R2ScoreRidge = r2_score(z, zRidgemean)
BiasRidge = Frank.Bias()[1]
MSERidge = Frank.MSE()[1]
VarRidge = Frank.Variance()[1]

#Lasso evaluated Franke function
beta_meanLasso /= Numbfolds
zLassomean = np.dot(X, beta_meanLasso)
R2ScoreLasso = r2_score(z, zLassomean)
BiasLasso = Frank.Bias()[1]
MSELasso = Frank.MSE()[1]
VarLasso = Frank.Variance()[1]



#Plotting
fig = plt.figure()
ax = fig.gca(projection= "3d")


#Plot the surface.
Frank = ax.plot_surface(x, y, z.reshape((n, n)), cmap=cm.coolwarm, linewidth=0, antialiased=False)

#Customize the z axis.
ax.set_zlim(-0.10, 1.40)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))
#Add a color bar which maps values to colors.
fig.colorbar(Frank, shrink=0.5, aspect=5)
#plt.show()"""
