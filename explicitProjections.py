"""
Analyse the explicit projections calculated
"""


from mpl_toolkits.mplot3d import Axes3D
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata

directory = "D:/uni/tomography-calibration/solid-angle-calculation/"

sangles = np.load(directory + "solid-angle-top-and-out-sensors-1-and-16.npy")
coords = np.load(directory + "out-coordinates-sensor-1.npy")

sangles = sangles[0:len(coords)]

# Summing over zz
# gridx,gridy=np.mgrid[-100:101,1:205].reshape(2,(101+100)*(205-1))

proj_values = []
summed_proj = []

gridx = np.arange(-100, 100.5, 0.5)
gridy = np.arange(-80, 80.5, 0.5)

for x in gridx:
    proj_values.append([])
    for y in gridy:
        proj_values[-1].append(
            np.sum(sangles[(np.abs(coords[:, 0] - x) < 0.0001) & (np.abs(coords[:, 1] - y) < 0.0001)]))

# %%

summed_proj = np.array(proj_values)

for column in summed_proj:

    column[np.argmax(column)] = np.sum(column)
    print(np.sum(column))


plot_gridx = np.arange(-100.25, 100.5, 0.5)
plot_gridy = np.arange(-80.25, 80.5, 0.5)

plt.figure()
plt.pcolormesh(plot_gridx, plot_gridy, np.array(proj_values).T)
plt.colorbar()
plt.xlabel("x (mm)")
plt.ylabel("y (mm)")
plt.show()

plt.figure()
plt.pcolormesh(plot_gridx, plot_gridy, summed_proj.T)
plt.colorbar()
plt.xlabel("x (mm)")
plt.ylabel("y (mm)")
plt.show()
# gridx,gridy,gridz=np.mgrid[-100:101:20,0:200:5,0:70:1]
#
# values=griddata(coords,sangles,(gridx,gridy,gridz),method='linear',fill_value=0.0)
##
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
##x = coords[:,0]
##y = coords[:,1]
##z = coords[:,2]
##c = sangles
#
# x = gridx.flatten()
# y = gridy.flatten()
# z = gridz.flatten()
# c = values.flatten()
#
# x = x[c>0]
# y = y[c>0]
# z = z[c>0]
# c = c[c>0]
#
# ax.scatter(x, y, z, c=c, cmap=plt.hot())
# plt.show()
