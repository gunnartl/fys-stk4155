import reg_class
from reg_class import*

iters = 100

betas  = np.zeros((3,iters,21))
#R2_train = np.zeros_like(R2_test)
#R2_vec  = np.zeros_like(MSE_vec)
deg = 5
n = 50
for j in range(iters):
    print(j)
    x = np.sort((np.random.rand(n)))
    y = np.sort((np.random.rand(n)))
    
    x,y = np.meshgrid(x,y)
    

    z = reg_class.FRANK(x.ravel(),y.ravel()) + 0.5*np.random.randn(n*n,)
    X = polynomial_this(x.ravel(),y.ravel(),deg)


    FRANK = regression(X,z)
    betas[0,j,:] = FRANK.OLS()
    
    betas[1,j,:] = FRANK.ridge(0.0005)

    betas[2,j,:] = FRANK.lasso(0.0005)

#%%
variance = np.var(betas,axis=1)
SD = np.sqrt(variance)
mean = np.mean(betas, axis=1,)
print(mean)
meanSD = np.mean(SD,axis=1)
print(meanSD)

