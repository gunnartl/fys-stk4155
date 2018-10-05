import reg_class
from reg_class import*

MSE_vec = np.zeros((3,13,200))
R2_vec  = np.zeros_like(MSE_vec)
lambdas   = 10.**np.array(range(-7,6))
for i in range(len(lambdas)):
    print(i)
    for j in range(10):
        n = 50
        x = np.sort((np.random.rand(n)))
        y = np.sort((np.random.rand(n)))
        
        x,y = np.meshgrid(x,y)
        
        deg = 5
        z = reg_class.FRANK(x.ravel(),y.ravel()) #+0.3*np.random.randn(n*n,)
        X = polynomial_this(x,y,deg)
        from sklearn.model_selection import train_test_split
        X,X_test,z,z_test = train_test_split(X, z, test_size = 0.2)

        FRANK = regression(X,z)
        #OLS_beta = FRANK.OLS()
        #MSE_vec[0,i,j] = MSE(z_test,X_test.dot(OLS_beta))

        ridge_beta = FRANK.ridge(lambdas[i])
        R2_vec[1,i,j] = R2(z_test,X_test.dot(ridge_beta))
        #MSE_vec[1,i,j] = FRANK.R2()

        lasso_beta = FRANK.lasso(lambdas[i])
        R2_vec[2,i,j] = R2(z_test,X_test.dot(lasso_beta))
        
        
#%%

MSE_mean = np.mean(MSE_vec,axis=2)
R2_mean  = np.mean(R2_vec,axis=2)
var = np.var(MSE_vec, axis=2)

import matplotlib.pyplot as plt
#plt.plot(MSE_mean[0])
plt.plot(lambdas,R2_mean[1])
plt.plot(lambdas,R2_mean[2])
#plt.legend(["OLS","RIDGE","lasso"])
plt.semilogx()
plt.show()


"""
lambdas = [10e-4 , 10e-3, 10e-2, 10e-1, 10e0, 10e1, 10e2, 10e3, 10e4]
def R2lambda(dataset1, dataset2, lambdas):
    Franklamb = Reg2.regression(dataset1, dataset2)
    R2ridge = []
    R2lasso = []
    R2OLS = []
    for lambd in lambdas:
        betasR = Franklamb.ridge(lambd)
        zRidge = np.dot(X, betasR)
        R2ridge.append(R2(z, zRidge))
        
        betasL = Franklamb.lasso(lambd)
        zLasso = np.dot(X, betasL)
        R2lasso.append(R2(z, zLasso))
        
        betasOLS = Franklamb.OLS()
        zOLS = np.dot(X, betasOLS)
        R2OLS.append(R2(z, zOLS))
    return R2OLS, R2ridge, R2lasso

R2OLSTrain, R2ridgeTrain, R2lassoTrain = R2lambda(X, z, lambdas)
R2OLSTest, R2ridgeTest, R2lassoTest = R2lambda(X_test, z_test, lambdas)
 
plt.plot(np.log10(lambdas),R2ridgeTest, '-' ,  label = 'Ridge Test', color = 'red')
plt.plot(np.log10(lambdas),R2OLSTest, '-' ,  label = 'OLS Test', color = 'blue')
plt.plot(np.log10(lambdas), R2lassoTest, '-' ,  label = 'Lasso Test', color = 'green')
plt.plot(np.log10(lambdas),R2ridgeTrain, '--' ,  label = 'Ridge Train', color = 'red')
plt.plot(np.log10(lambdas),R2OLSTrain, '--' ,  label = 'OLS Train', color = 'blue')
plt.plot(np.log10(lambdas), R2lassoTrain, '--' ,  label = 'Lasso Train', color = 'green')
plt.xlabel(r'$log_{10}(\lambda)$', fontsize = 16)
plt.ylabel(r'R2', fontsize = 16)
plt.legend()"""
        