import reg_class
from reg_class import*
from sklearn.model_selection import train_test_split

iters = 1000

betas  = np.zeros((3,iters,21))
#R2_train = np.zeros_like(R2_test)
#R2_vec  = np.zeros_like(MSE_vec)
deg = 5
n = 50
for j in range(iters):

    x = np.sort((np.random.rand(n)))
    y = np.sort((np.random.rand(n)))
    
    x,y = np.meshgrid(x,y)
    
    x = x.ravel()
    y = y.ravel()

    z = reg_class.FRANK(x,y + 0.5*np.random.randn(n*n))
    X = polynomial_this(x,y,deg)

    FRANK = regression(X,z)
    
    betas[0,j,:] = FRANK.OLS()
    
    betas[1,j,:] = FRANK.ridge(0.0005)

    betas[2,j,:] = FRANK.lasso(0.0005)

#%%
sigma2 = (1./(len(z)-X.shape[1]-1))*sum((z-X.dot(FRANK.OLS()))**2)
covar = np.linalg.inv(X.T.dot(X))*sigma2
var = np.diagonal(covar)
print(var)

    
variance = np.var(betas,axis=1)
SD = np.sqrt(variance)
mean = np.mean(betas, axis=1,)
print(mean)
meanSD = np.mean(SD,axis=1)

