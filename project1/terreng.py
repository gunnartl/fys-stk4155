from imageio import imread
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import reg_class
from reg_class import *
from sklearn.model_selection import train_test_split

# Load the terrain
terrainfull = np.array(imread("SRTM_data_Norway_1.tif"))
terrain1 = terrainfull[1000:1200,800:1100]
tx = np.linspace(0,1,terrain1.shape[0])
ty = np.linspace(0,1,terrain1.shape[1])
tx,ty = np.meshgrid(tx,ty)
tx1d = tx.ravel()
ty1d = ty.ravel()
terrain11d = terrain1.ravel()

deg = 5
TX = polynomial_this(tx1d,ty1d,deg)

terreng = regression(TX, terrain11d)
beta_OLS = terreng.OLS()
zterreng = TX.dot(beta_OLS) 
Numbdeg = 20
Variterate = np.zeros(Numbdeg)
Biasiterate = np.zeros(Numbdeg)
MSEiterate = np.zeros(Numbdeg)
Numbfolds = 10

Xterr, Xterr_holdout, zterr, zterr_holdout = train_test_split(TX, zterreng, test_size = 0.2, random_state = 40)
zterr = zterr[:,np.newaxis]
xterr_holdout = Xterr_holdout[:, 1]
yterr_holdout = Xterr_holdout[:, 2]
X_Split, z_Split = KfoldCrossVal(Xterr, zterr, Numbfolds)

z_mean = np.zeros((len(Xterr_holdout), Numbfolds, Numbdeg))
zterr_holdout = zterr_holdout[:, np.newaxis]

for i in range(Numbfolds):
    XTrainSets = np.delete(X_Split, i, 0)
    XTestSets = X_Split[i]
    zTrainSets = np.delete(z_Split, i, 0)
    zTestSets = z_Split[i]
    XTrainSets = np.vstack(XTrainSets)
    zTrainSets = np.vstack(zTrainSets)
    for k in range(1, Numbdeg):
        Xiterate = polynomial_this(XTrainSets[:,1], XTrainSets[:,2], k)
        Frankiterate = regression(Xiterate, zTrainSets)
        betas = Frankiterate.OLS()
        X_holdout_k = polynomial_this(xterr_holdout, yterr_holdout, k)
        zPred = np.dot(X_holdout_k, betas)
        z_mean[:, i, k] = list(zPred)

errorterr = np.zeros(Numbdeg - 1)
Biasterr  = np.zeros(Numbdeg - 1)
Varianceterr = np.zeros(Numbdeg - 1)

for k in range(len(Xterr_holdout)):
    for j in range(Numbdeg - 1):
        errorterr[j] = np.mean(MSE(zterr_holdout, z_mean[k, :, j])) 
        Biasterr[j] = bias(zterr_holdout, z_mean[k, :, j])
        Varianceterr[j] = variance(z_mean[k, :, j])


plt.plot(np.linspace(1, Numbdeg - 1, Numbdeg - 1), Varianceterr, '--', label = 'Var')
plt.plot(np.linspace(1, Numbdeg - 1, Numbdeg - 1), Biasterr, '--', label = 'Bias^2')
plt.plot(np.linspace(1, Numbdeg - 1, Numbdeg - 1), errorterr, '--', label = 'MSE')
plt.legend()
plt.show()

#beta_OLS = terreng.OLS()
#pred_OLS = TX.dot(beta_OLS)
#pred_OLS = pred_OLS.reshape((terrain1.shape[0],terrain1.shape[1]))
#
#beta_ridge = terreng.ridge(0.001)
#pred_ridge = TX.dot(beta_ridge)
#pred_ridge = pred_ridge.reshape((terrain1.shape[0],terrain1.shape[1]))
#
#beta_lasso = terreng.lasso(0.0001)
#pred_lasso = TX.dot(beta_lasso)
#pred_lasso = pred_lasso.reshape((terrain1.shape[0], terrain1.shape[1]))

# Show the terrain
from matplotlib.ticker import LinearLocator, FormatStrFormatter
fig = plt.figure()
ax = fig.gca(projection="3d")

# Plot the surface.
surf = ax.plot_surface(-tx,-ty,terrain1, cmap=cm.coolwarm,
linewidth=0, antialiased=False)
# Customize the z axis.
#ax.set_zlim(-0.10, 1.40)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))
# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.hold("on")

#plt.show()
plt.figure()
plt.title("Terrain_over_Norway_1")
plt.imshow(terrain1, cmap="gray")
plt.xlabel("X")
plt.ylabel("Y")
plt.hold("on")
#plt.show()

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
from matplotlib import cm
fig = plt.figure()
ax = fig.gca(projection="3d")
# Plot the surface
surf = ax.plot_surface(-tx,-ty, pred_ridge, cmap=cm.coolwarm,
linewidth=0, antialiased=False)
# Customize the z axis.
#ax.set_zlim(-0.10, 1.40)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))
# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()