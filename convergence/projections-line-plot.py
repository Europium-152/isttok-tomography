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

cone_projections_dic = load_projection("complex-view-cone-80.npy")[0]
cone_projections = cone_projections_dic['projections']
cone_x_grid = cone_projections_dic['x']
cone_y_grid = cone_projections_dic['y']

line_projections_dic = load_projection("line-approximation-80.npy")[0]
line_projections = line_projections_dic['projections']
line_x_grid = line_projections_dic['x']
line_y_grid = line_projections_dic['y']


# Plot projections cone and line together ------------------------------------------------------------------------------

i = 22
j = 25
summed_projections = line_projections[i]*0.001 + cone_projections[j]
plt.figure(constrained_layout=True)
plt.axis(xmin=0, xmax=85, ymin=-85, ymax=85, option='off')
plt.pcolormesh(cone_x_grid - 0.5 * (cone_x_grid[1] - cone_x_grid[0]),
               cone_y_grid + 0.5 * (cone_y_grid[0] - cone_y_grid[1]),
               summed_projections,
               vmin=None,
               vmax=0.005)
# plt.pcolormesh(x_grid, y_grid, summed_projections)
plt.axis('scaled')
plt.axis(xmin=-85, xmax=85, ymin=-85, ymax=85)
cbar = plt.colorbar()
cbar.set_label("\nLine length inside pixel (m) / Angle subtended (rad)")
plt.xlabel("R (mm)")
plt.ylabel("z (mm)")
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


# summed_projections = 0.
# projection_indices = np.arange(16, 32, 2)
# projection_indices = [23]
# # for i in range(len(projections)):
# for i in projection_indices:
#     summed_projections += projections[i]
#
# plt.figure(constrained_layout=True)
# plt.axis(xmin=0, xmax=85, ymin=-85, ymax=85, option='off')
# plt.pcolormesh(x_grid - 0.5 * (x_grid[1] - x_grid[0]),
#                y_grid + 0.5 * (y_grid[0] - y_grid[1]),
#                summed_projections,
#                vmin=None,
#                vmax=None)
# # plt.pcolormesh(x_grid, y_grid, summed_projections)
# plt.axis('scaled')
# plt.axis(xmin=-85, xmax=85, ymin=-85, ymax=85)
# cbar = plt.colorbar()
# cbar.set_label("Line length inside pixel (mm)")
# plt.xlabel("R (mm)")
# plt.ylabel("z (mm)")
