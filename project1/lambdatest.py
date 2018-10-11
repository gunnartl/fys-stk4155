import reg_class
from reg_class import*
from sklearn.model_selection import train_test_split

"""
Gir et lambda - r2 plott av OLS, rigde og lasso. Med noise = 0 gir det to 
nesten overlappende kurver for test og train dataene og med noise f.eks= 0.5
gir det litt dårlige resultater for testdataene

De stiplede liniene i plottet er Testdataene , de heltrukene er Treningsdataene
"""
Noise = .5
iters = 50

lambdas   = 10.**np.array(range(-7,10))

R2_test  = np.zeros((3,len(lambdas),iters))
R2_train = np.zeros_like(R2_test)

MSE_test = np.zeros_like(R2_test)
MSE_train = np.zeros_like(R2_test)

lasso_betas = np.zeros((len(lambdas),iters,21))
ridge_betas = np.zeros_like(lasso_betas)

#R2_vec  = np.zeros_like(MSE_vec)
n = 100
deg = 5
for i in range(len(lambdas)):
    print(i)
    for j in range(iters):
        x = np.sort((np.random.rand(n)))
        y = np.sort((np.random.rand(n)))
        
        x,y = np.meshgrid(x,y)
        
        z = reg_class.FRANK(x.ravel(),y.ravel()) +Noise*np.random.randn(n*n,)
        X = polynomial_this(x.ravel(),y.ravel(),deg)

        X,X_test,z,z_test = train_test_split(X, z, test_size = 0.2)

        FRANK = regression(X,z)
        OLS_beta = FRANK.OLS()
        R2_test[0,i,j] = R2(z_test,X_test.dot(OLS_beta))
        R2_train[0,i,j] = R2(z,X.dot(OLS_beta))
        MSE_test[0,i,j] = MSE(z_test,X_test.dot(OLS_beta))
        MSE_train[0,i,j] = MSE(z,X.dot(OLS_beta))        

        
        ridge_beta = FRANK.ridge(lambdas[i])
        ridge_betas[i,j,:] = ridge_beta
        R2_test[1,i,j] = R2(z_test,X_test.dot(ridge_beta))
        R2_train[1,i,j] = R2(z,X.dot(ridge_beta))
        MSE_test[1,i,j] = MSE(z_test,X_test.dot(ridge_beta))
        MSE_train[1,i,j] = MSE(z,X.dot(ridge_beta))

        lasso_beta = FRANK.lasso(lambdas[i]).squeeze()
        lasso_betas[i,j,:] = lasso_beta
        R2_test[2,i,j] = R2(z_test,X_test.dot(lasso_beta))
        R2_train[2,i,j] = R2(z,X.dot(lasso_beta))
        MSE_test[2,i,j] = MSE(z_test,X_test.dot(lasso_beta))
        MSE_train[2,i,j] = MSE(z,X.dot(lasso_beta))
        
#%%

R2_mean_test  = np.mean(R2_test,axis=2)
R2_mean_train  = np.mean(R2_train,axis=2)

MSE_mean_test  = np.mean(MSE_test,axis=2)
MSE_mean_train  = np.mean(MSE_train,axis=2)

# plotter R2
import matplotlib.pyplot as plt
plt.plot(lambdas,R2_mean_train[0],"orange")
plt.plot(lambdas,R2_mean_train[1],"steelblue")
plt.plot(lambdas,R2_mean_train[2],"yellowgreen")

plt.plot(lambdas,R2_mean_test[0],"orange",linestyle="-.")
plt.plot(lambdas,R2_mean_test[1],"steelblue",linestyle="-.")
plt.plot(lambdas,R2_mean_test[2],"yellowgreen",linestyle="-.")
plt.legend(["OLS","RIDGE","lasso"])
plt.semilogx()
plt.title("R2 as a function of Lambda for a rather noisy Franke-function")
plt.xlabel("$\lambda$")
plt.ylabel("R2")
plt.show()


#plotter mse-grafer
plt.plot(lambdas,MSE_mean_train[0],"orange")
plt.plot(lambdas,MSE_mean_train[1],"steelblue")
plt.plot(lambdas,MSE_mean_train[2],"yellowgreen")

plt.plot(lambdas,MSE_mean_test[0],"orange",linestyle="-.")
plt.plot(lambdas,MSE_mean_test[1],"steelblue",linestyle="-.")
plt.plot(lambdas,MSE_mean_test[2],"yellowgreen",linestyle="-.")
plt.legend(["OLS","RIDGE","lasso"])
plt.semilogx()
plt.title("MSE as a function of Lambda for a rather noisy Franke-function")
plt.xlabel("$\lambda$")
plt.ylabel("MSE")
plt.show()
#%%
print(lasso_betas.shape)
lasso_betas = np.mean(lasso_betas,axis=1)
ridge_betas = np.mean(ridge_betas,axis=1)
print(lasso_betas[:,0].shape)
plt.hold("on")
#%%
for i in range(21):
    plt.plot(lambdas[:10],ridge_betas[:10,i])
plt.title("Development in size of coefficients for Ridge as a funtion of $\lambda$")
plt.xlabel("$\lambda$")
plt.ylabel("Size of coefs")
plt.semilogx()
plt.show()

for i in range(21):
    plt.plot(lambdas[:10],lasso_betas[:10,i])
plt.title("Development in size of coefficients for Lasso as a funtion of $\lambda$")
plt.xlabel("$\lambda$")
plt.ylabel("Size of coefs")
plt.semilogx()
plt.show()





