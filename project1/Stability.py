import reg_class
from reg_class import*
from imageio import imread

terrainfull = np.array(imread("SRTM_data_Norway_1.tif"))
MSE_vec = np.zeros((3,20,200))
R2_vec  = np.zeros_like(MSE)
n_vec = []
for i in range(20):
    print(i)
    n = (i+13)
    n_vec.append(n)
    for j in range(200):

        #z = terrainfull[1000:1000+n,800:800+n]
        #tx = np.linspace(0,1,z.shape[0])
        #ty = np.linspace(0,1,z.shape[1])
        
        #tx,ty = np.meshgrid(tx,ty)
        
        #tx1d = tx.ravel()
        #ty1d = ty.ravel()
        #terrain11d=z.ravel()
        
        #deg = 5
        #z = z.ravel()
        #X = polynomial_this(tx1d,ty1d,deg)
        
        
    
        x = np.sort((np.random.rand(n)))
        y = np.sort((np.random.rand(n)))
        
        x,y = np.meshgrid(x,y)
        
        deg = 5
        z = reg_class.FRANK(x.ravel(),y.ravel()) +0.3*np.random.randn(n*n,)
        X = polynomial_this(x.ravel(),y.ravel(),deg)
        from sklearn.model_selection import train_test_split
        X,X_test,z,z_test = train_test_split(X, z, test_size = 0.2)
        
        FRANK = regression(X,z)
        OLS_beta = FRANK.OLS()
        MSE_vec[0,i,j] = MSE(z_test,X_test.dot(OLS_beta))

        ridge_beta = FRANK.ridge(0.01)
        MSE_vec[1,i,j] = MSE(z_test,X_test.dot(ridge_beta))


        lasso_beta = FRANK.lasso(0.01)
        MSE_vec[2,i,j] = MSE(z_test,X_test.dot(lasso_beta))
        

#%%
MSE_mean = np.mean(MSE_vec,axis=2)
SD = np.sqrt(np.var(MSE_vec, axis=2))/2
n = np.array(range(10))*10
import matplotlib.pyplot as plt
plt.plot(n_vec,MSE_mean[0])
plt.errorbar(n_vec,MSE_mean[0],SD[0],errorevery=4,fmt = 'o')
plt.plot(n_vec,MSE_mean[1])
plt.errorbar(n_vec,MSE_mean[1],SD[1],errorevery=3,fmt = 'o')
plt.plot(n_vec,MSE_mean[2])
plt.errorbar(n_vec,MSE_mean[2],SD[2],errorevery=2,fmt = 'o')
plt.legend(["OLS","RIDGE","lasso"])
#plt.semilogy()
plt.show()

        