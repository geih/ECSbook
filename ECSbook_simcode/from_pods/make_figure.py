import h5py
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from ECSbook_simcode import plotting_convention
from mpl_toolkits.mplot3d import Axes3D

f = h5py.File("output-00600.h5", 'r')

# for i in f.keys():
#     print(i, type(i))

x = f['DOMAIN']['x']
y = f['DOMAIN']['y']
pot = f['DOMAIN']['pot']


# Use 80 of 100 points in x-direction
xrange = range(0,100)
# Use part of y-range immediately above the membrane (index 32)
yrange = range(31,45)


# Attention: x- and y-dimensions in HDF5 files are swapped!
# y-indices come at before x-indices, contrary to common convention
X = np.array(x)[yrange,:][:,xrange]
Y = np.array(y)[yrange,:][:,xrange]
Z = np.array(pot)[yrange,:][:,xrange] * 1000

# cmap = cm.coolwarm
cmap = cm.bwr

fig = plt.figure(figsize=(10, 5))

membrane_thickness = 5  # nm
axon_radius = 500  # nm
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X/1e3, Y/1e3, Z, cmap=cmap)
ax.set_xlabel("x [µm]")
ax.set_ylabel("y [µm]")
ax.set_yticks(np.arange(0.505, 0.515, 0.003))
ax.set_zlabel("pot [µV]")
ax.azim = -100
plt.savefig("fig_PNP_3D.png")

plt.close(fig)

fig = plt.figure(figsize=(7, 5))
fig.subplots_adjust(right=0.85, left=0.12)
vmax = np.max(np.abs(Z))
# print(vmax)
levels = np.linspace(-vmax, vmax, 32)

ax = fig.add_subplot(111, xlim=[2500000, 7500000], ylim=[-1, 6],
                     xlabel="position along axon (nm)",
                     ylabel="distance from\nmembrane surface (nm)")
img = ax.contourf(X, Y - 505, Z, levels=levels, cmap=cmap,
                  vmax=vmax, vmin=-vmax)
ax.contour(X, Y - 505, Z, levels=levels, colors='k', vmax=vmax, vmin=-vmax)


ax.axhspan(axon_radius - 505, axon_radius + membrane_thickness - 505,
           facecolor='0.8')
ax.text(2600000, -0.1, "axon membrane", va="top")
plotting_convention.simplify_axes(ax)

ax.set_xticks([3e6, 4e6, 5e6, 6e6, 7e6])
ax.set_xticklabels(["3$\cdot10^6$", "4$\cdot10^6$",
                    "5$\cdot10^6$", "6$\cdot10^6$", "7$\cdot10^6$"])
cax = fig.add_axes([0.87, 0.15, 0.01, 0.7])
plt.colorbar(img, cax=cax, ticks=[-200, -100, 0, 100, 200], label="µV")
plt.savefig("fig_PNP.png")


# plt.show()