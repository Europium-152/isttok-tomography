import sys

sys.path.append("../")

from projectionSelector import load_projection
import numpy as np
import matplotlib.pyplot as plt
from tomography import MFI
import matplotlib.style as mplstyle
mplstyle.use('ggplot')

# plot = False
plot = True

# Load projection matrix ---------------------------------------------------------------------------------

projections_dic = load_projection("complex-view-cone-45.npy")[0]
projections = projections_dic['projections']
x_grid = projections_dic['x']
y_grid = projections_dic['y']

# HD projection matrix -----------------------------------------------------------------------------------

projections_dicHD = load_projection("complex-view-cone-100.npy")[0]
projectionsHD = projections_dicHD['projections']
x_gridHD = projections_dicHD['x']
y_gridHD = projections_dicHD['y']


# Plot projections ---------------------------------------------------------------------------------------

# for i in (0, 16, 32):
#
#     rows = 4
#     cols = 4
#
#     fig, axes = plt.subplots(nrows=rows, ncols=cols)
#
#     for ax, p in zip(axes.flat, projections[i:i + 16]):
#         ax.set_aspect('equal', 'datalim')
#         im = ax.pcolormesh(mfi.x_array_plot, mfi.y_array_plot, p, vmin=None, vmax=None)
#         circle = plt.Circle((0., 0.), 85., color='w', fill=False)
#         ax.add_artist(circle)
#
#     fig.subplots_adjust(right=0.8)
#     cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
#     fig.colorbar(im, cax=cbar_ax)
#
#     plt.show()

# Plot the sum of the projections -----------------------------------------------------------------------


summed_projections = 0.
projection_indices = [0, 3]
# for i in range(len(projections)):
for i in projection_indices:
    summed_projections += projections[i]

plt.figure(constrained_layout=True)
plt.axis(xmin=0, xmax=85, ymin=-85, ymax=85, option='off')
plt.pcolormesh(x_grid - 0.5 * (x_grid[1] - x_grid[0]),
               y_grid + 0.5 * (y_grid[0] - y_grid[1]),
               summed_projections)
# plt.pcolormesh(x_grid, y_grid, summed_projections)
plt.axis('scaled')
plt.axis(xmin=0, xmax=85, ymin=-85, ymax=85)
plt.colorbar()

# Plot contour lines around the cones of sight ---------------------------------------------------------

contour_colors = ['white', 'white']
linestyles = ['solid', 'dashed']
for i, color, line in zip(projection_indices, contour_colors, linestyles):
    plt.contour(x_gridHD, y_gridHD, projectionsHD[i], levels=[0.0001], colors=color, linestyles=line)

# plt.tight_layout(0.)
plt.show()

