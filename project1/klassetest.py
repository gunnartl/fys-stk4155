#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from sklearn.linear_model import LinearRegression
from numba import jit
from reg_class import regression,polynomial_this
from sklearn import linear_model
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

from imageio import imread

# Load the terrain
terrainfull = np.array(imread("SRTM_data_Norway_1.tif"))
terrain1 = terrainfull#[1000:2000,800:1800]
tx = np.linspace(0,1,terrain1.shape[0])
ty = np.linspace(0,1,terrain1.shape[1])

tx,ty = np.meshgrid(tx,ty)

tx1d = tx.ravel()
ty1d = ty.ravel()
terrain11d=terrain1.ravel()

deg = 5

TX = polynomial_this(tx1d,ty1d,deg)

terreng = regression(TX,terrain11d)
terreng_beta = terreng.ridge(0.001)
print(terreng_beta)
TX = polynomial_this(tx1d,ty1d,deg)
terengplutt = TX.dot(terreng_beta)
terengplutt = terengplutt.reshape((terrain1.shape[0],terrain1.shape[1]))

print(terreng.R2(),"THOMMASERBEST")
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
#oppg 1"""


from sklearn.metrics import r2_score

#print(max(z-temp))

print(r2_score(z, frankreg.plot().ravel()),"sklearn")

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
from matplotlib import cm
fig = plt.figure()
ax = fig.gca(projection="3d")
# Plot the surface.
surf = ax.plot_surface(-tx,-ty,terengplutt, cmap=cm.coolwarm,
linewidth=0, antialiased=False)
# Customize the z axis.
#ax.set_zlim(-0.10, 1.40)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))
# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()