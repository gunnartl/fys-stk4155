import reg_class
from reg_class import*

MSE_vec = np.zeros((3,20,200))
R2_vec  = np.zeros_like(MSE)

for i in range(20):
    print(i)
    for j in range(100):
        n = (i+13)
        x = np.sort((np.random.rand(n)))
        y = np.sort((np.random.rand(n)))
        
        x,y = np.meshgrid(x,y)
        
        deg = 5
        z = reg_class.FRANK(x.ravel(),y.ravel()) +0.3*np.random.randn(n*n,)
        X = polynomial_this(x,y,deg)
        from sklearn.model_selection import train_test_split
        X,X_test,z,z_test = train_test_split(X, z, test_size = 0.2)
        
        FRANK = regression(X,z)
        OLS_beta = FRANK.OLS()
        MSE_vec[0,i,j] = MSE(z_test,X_test.dot(OLS_beta))

        ridge_beta = FRANK.ridge(0.01)
        MSE_vec[1,i,j] = MSE(z_test,X_test.dot(ridge_beta))


        lasso_beta = FRANK.lasso(0.01)
        MSE_vec[2,i,j] = MSE(z_test,X_test.dot(lasso_beta))
        


MSE_mean = np.mean(MSE_vec,axis=2)
var = np.var(MSE_vec, axis=2)
n = np.array(range(10))*10
import matplotlib.pyplot as plt
plt.plot(MSE_mean[0])
plt.plot(MSE_mean[1])
plt.plot(MSE_mean[2])
plt.legend(["OLS","RIDGE","lasso"])
#plt.semilogy()
plt.show()

        