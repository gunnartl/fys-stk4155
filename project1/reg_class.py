import numpy as np
from sklearn import linear_model
from numba import jit
#from sklearn.linear_model import LinearRegression

@jit
def polynomial_this(x,y,n):
    """
    Lager designmatrise for for en funksjon av to variabler
    for alle polynomer opp til og med grad n
    """
    X = np.c_[np.ones(len(x))]
    for i in range(1,n+1):
        X = np.c_[X,x**(i)]
        for j in range(i-1,0,-1):
            X = np.c_[X,(x**(j))*(y**(i-j))]  
        X = np.c_[X,y**(i)]
    return X

def bias(true, pred):
    bias = np.mean((true - np.mean(pred))**2)
    return bias

    
def MSE(true, pred):
    MSE = sum((true-pred)**2)/(len(true))
    return MSE
    
def R2(true, pred):
    R2 = 1-(np.sum((true - pred)**2)/np.sum((true-np.mean(pred))**2))
    return R2

def KfoldCrossVal(dataset, dataset2, Numbfold):
    """
    Takes in two coupled datasets and returns them splitted into k-matching 
    "folds" by produsing randomindices. 
    
    KfoldCrosVal([1,4,5,6],[7,6,8,5],2) may return
    
    [[1,5],[4,6]], [[7,8],[6,2]]
    
    by produsing the indices [[0,2],[1,3]] from input "dataset"
    """
    indices = np.arange(len(dataset[:, 0]))
    random_indices = np.random.choice(indices, size = len(dataset[:, 0]), replace = False)
    interval = int(len(dataset[:, 0])/Numbfold)
    datasetsplit = []
    dataset2split = []
    for k in range(Numbfold):
        datasetsplit.append(dataset[random_indices[interval*k : interval*(k + 1)]]) 
        dataset2split.append(dataset2[random_indices[interval*k : interval*(k + 1)]])

    return np.asarray(datasetsplit), np.asarray(dataset2split) 


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
        beta = lasso.coef_[:,np.newaxis]
        self.znew = self.X.dot(beta)
        return beta

         
    def beata_variance(self):
        sigma2 = (1./(len(self.z)-self.X.shape[1]-1))*sum((self.z-self.znew)**2)
        covar = np.linalg.inv(self.X.T.dot(self.X))*sigma2
        var = np.diagonal(covar)
        return beta_var

    
    def MSE(self):
        MSE = np.mean((self.z-self.znew)**2)
        return MSE
    
    def R2(self):
        self.R2 = 1-(np.sum((self.z - self.znew)**2)/np.sum((self.z-np.mean(self.z))**2))
        return self.R2


def FRANK(x, y):
    """
    Frankie function for testing the class
    """
    term1 = 0.75*np.exp(-(0.25*(9*x - 2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x + 1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4    
    


if __name__== "__main__" :
    def test_reg():
        a = 3
        
    