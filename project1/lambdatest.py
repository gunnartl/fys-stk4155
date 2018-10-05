import reg_class
from reg_class import*

R2_test  = np.zeros((3,13,50))
R2_train = np.zeros_like(R2_test)
#R2_vec  = np.zeros_like(MSE_vec)
lambdas   = 10.**np.array(range(-7,6))
for i in range(len(lambdas)):
    print(i)
    for j in range(50):
        n = 50
        x = np.sort((np.random.rand(n)))
        y = np.sort((np.random.rand(n)))
        
        x,y = np.meshgrid(x,y)
        
        deg = 5
        z = reg_class.FRANK(x.ravel(),y.ravel()) +0.5*np.random.randn(n*n,)
        X = polynomial_this(x.ravel(),y.ravel(),deg)
        from sklearn.model_selection import train_test_split
        X,X_test,z,z_test = train_test_split(X, z, test_size = 0.3)

        FRANK = regression(X,z)
        OLS_beta = FRANK.OLS()
        R2_test[0,i,j] = R2(z_test,X_test.dot(OLS_beta))
        R2_train[0,i,j] = R2(z,X.dot(OLS_beta))
        
        ridge_beta = FRANK.ridge(lambdas[i])
        R2_test[1,i,j] = R2(z_test,X_test.dot(ridge_beta))
        R2_train[1,i,j] = R2(z,X.dot(ridge_beta))

        lasso_beta = FRANK.lasso(lambdas[i])
        R2_test[2,i,j] = R2(z_test,X_test.dot(lasso_beta))
        R2_train[2,i,j] = R2(z,X.dot(lasso_beta))
        
#%%

R2_mean_test  = np.mean(R2_test,axis=2)
R2_mean_train  = np.mean(R2_train,axis=2)
#var = np.var(MSE_vec, axis=2)
print(np.mean(R2_mean_test-R2_mean_train))
import matplotlib.pyplot as plt
plt.plot(lambdas,R2_mean_train[0],"orange")
plt.plot(lambdas,R2_mean_train[1],"steelblue")
plt.plot(lambdas,R2_mean_train[2],"yellowgreen")

plt.plot(lambdas,R2_mean_test[0],"orange",linestyle="-.")
plt.plot(lambdas,R2_mean_test[1],"steelblue",linestyle="-.")
plt.plot(lambdas,R2_mean_test[2],"yellowgreen",linestyle="-.")
plt.legend(["OLS","RIDGE","lasso"])
plt.semilogx()
plt.xlabel("$\lambda$")
plt.ylabel("R2-score")
plt.show()