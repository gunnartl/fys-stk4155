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

class regression:
    def __init__(self,x,y,z,n,deg):
        self.x = x
        self.y = y
        self.z = z
        self.n = n
        self.deg = deg
        self.X = polynomial_this(x,y,deg)
        
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

         
    def beata_variance(self):
        sigma2 = (1./(len(self.z)-self.X.shape[1]-1))*sum((self.z-self.znew)**2)
        covar = np.linalg.inv(self.X.T.dot(self.X))*sigma2
        var = np.diagonal(covar)
        return beta_var
    
    def variance(self):
        var = np.mean((self.znew-np.mean(self.znew))**2)
        return var


    def plot(self):
        plutt = self.znew.reshape((self.n,self.n))
        x = self.x.reshape((self.n,self.n))
        y = self.y.reshape((self.n,self.n))
        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib.ticker import LinearLocator, FormatStrFormatter
        import matplotlib.pyplot as plt
        from matplotlib import cm
        fig = plt.figure()
        ax = fig.gca(projection="3d")
        # Plot the surface.
        surf = ax.plot_surface(x,y,plutt, cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)
        # Customize the z axis.
        #ax.set_zlim(-0.10, 1.40)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))
        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()
        return plutt
    
    def MSE(self):
        MSE = np.mean((self.z-self.znew)**2)
        return MSE
    
    def R2(self):
        self.R2 = 1-(np.sum((self.z - self.znew)**2)/np.sum((self.z-np.mean(self.z))**2))
        return self.R2
    def bias(self):
        bias = np.mean((self.z-np.mean(self.znew))**2)
        return bias
    def term(self):
        term = 2*sum((self.z-np.mean(self.znew))*(np.mean(self.znew)-self.znew))/len(self.z)
        return term