import numpy as np
from imageio import imread
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import reg_class
from reg_class import *
from sklearn.model_selection import train_test_split
# Load the terrain
#terrain1 = imread("yosemite.tif") ## Yosemite valley
#terrain1 = terrain1[750:950,1550:1900]
terrain1 = imread("everest.tif") ## Yosemite valley
terrain1 = terrain1[:150,3200:3400]

tx = np.linspace(0,1,terrain1.shape[1])
ty = np.linspace(0,1,terrain1.shape[0])

tx,ty = np.meshgrid(tx,ty)

tx1d = tx.ravel()
ty1d = ty.ravel()
terrain11d=terrain1.ravel()

#%% Plotting
plt.figure()
plt.title("Mount Everest")
plt.imshow(terrain1, cmap=cm.coolwarm)
plt.xlabel("X")
plt.ylabel("Y")
plt.show()


from matplotlib.ticker import LinearLocator, FormatStrFormatter
fig = plt.figure()
ax = fig.gca(projection="3d")
surf = ax.plot_surface(tx,-ty,terrain1.reshape(tx.shape), cmap="gray",
linewidth=0, antialiased=False)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.title("Mount Everest")
plt.show()
#%%
deg = 10
Numbfolds = 5

X_full = polynomial_this(tx1d,ty1d,deg)
z_full = terrain11d


X, X_holdout, z, z_holdout = train_test_split(X_full, z_full, test_size = 0.2)

X_Split, z_Split = KfoldCrossVal(X, z, Numbfolds)
print (X_Split.shape,z_Split.shape)
z_Split = z_Split[:,:,np.newaxis]
R2s  = np.zeros(Numbfolds)
MSEs = np.zeros(Numbfolds)

for i in range(Numbfolds):
    XTrainSets = np.delete(X_Split, i, 0)
    XTestSets = X_Split[i]
    zTrainSets = np.delete(z_Split, i, 0)
    zTestSets = z_Split[i]
    XTrainSets = np.vstack(XTrainSets)
    zTrainSets = np.vstack(zTrainSets)

    reg = regression(XTrainSets,zTrainSets)
    betas  = reg.OLS().squeeze()
    R2s[i] = R2(z_holdout,X_holdout.dot(betas))
    MSEs[i] = MSE(z_holdout,X_holdout.dot(betas))
    


print(R2s,MSEs)


endelig_modell = regression(X_full,z_full)
OLSbeta = endelig_modell.OLS()
plott = X_full.dot(OLSbeta).reshape(tx.shape)

print(endelig_modell.R2())
print(endelig_modell.MSE())


from matplotlib.ticker import LinearLocator, FormatStrFormatter
fig = plt.figure()
ax = fig.gca(projection="3d")
# Plot the surface.
surf = ax.plot_surface(tx,-ty,plott, cmap=cm.coolwarm,
linewidth=0, antialiased=False)
# Customize the z axis.
#ax.set_zlim(-0.10, 1.40)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))
# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)
#plt.hold("on")
plt.show()






