import numpy as np
from sklearn import linear_model
from numba import jit
#from sklearn.linear_model import LinearRegression

@jit
def polynomial_this(x,y,n):
    X = np.c_[np.ones(len(x))]
    for i in range(1,n+1):
        X = np.c_[X,x**(i)]
        for j in range(i-1,0,-1):
            X = np.c_[X,(x**(j))*(y**(i-j))]  
        X = np.c_[X,y**(i)]
    return X

def bias(true, pred):
    bias = np.mean((y_test - y_pred)**2)
    return bias


         
def variance(pred):
    var = np.var(pred)
    return var

    
def MSE(true, pred):
    MSE = sum((true-pred)**2)/(len(true))
    return MSE
    
def R2(true, pred):
    R2 = 1-(np.sum((true - pred)**2)/np.sum((true-np.mean(pred))**2))
    return R2

class regression:
    def __init__(self,X,z):
        self.z = z
        self.X = X
        
    @jit    
    def ridge(self,lambd):
        X = self.X
        beta = np.linalg.inv(X.T.dot(X)+lambd*np.identity(X.shape[1])).dot(X.T.dot(self.z))
        self.znew = X.dot(beta)

        self.beta = beta
        return beta#plutt
    
    @jit
    def OLS(self):
        X = self.X
        beta = np.linalg.inv(X.T.dot(X)).dot(X.T.dot(self.z))
        self.znew = X.dot(beta)
        return beta
    
    @jit
    def lasso(self, lambd):
        lasso = linear_model.Lasso(alpha = lambd,fit_intercept = False)
        lasso.fit(self.X, self.z)
        beta = lasso.coef_
        self.znew = self.X.dot(beta)
        return beta
