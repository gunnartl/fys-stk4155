import reg_class
from reg_class import*

n = 100
deg = 8

x = np.sort((np.random.rand(n)))
y = np.sort((np.random.rand(n)))

x, y = np.meshgrid(x,y)

x1d = x.reshape((n**2, 1))
y1d = y.reshape((n**2, 1))



noise = 0.5*np.random.randn(n**2, 1)

z       = FRANK(x1d,y1d)
z_noise = noicyfrank = z + noise

X = polynomial_this(x1d, y1d, deg)

frankreg = regression(X,z)
beta = frankreg.OLS()

frankreg_noise = regression(X,z_noise)
beta_noise     = frankreg_noise.OLS()

znew = X.dot(beta)
znew_noise = X.dot(beta_noise)

R2_score = R2(z,znew)
mse      = MSE(z,znew)

R2_score_noise = R2(z,znew_noise)
mse_noise      = MSE(z,znew_noise)

print ("R2 of OLS on full dataset=", R2_score)
print ("R2 of OLS on full dataset  with noise=", R2_score_noise)

print ("MSE of OLS on full dataset = ", mse)
print ("MSE of OLS on full dataset with noise = ", mse_noise)