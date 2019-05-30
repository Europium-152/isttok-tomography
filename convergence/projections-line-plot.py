import sys

sys.path.append("../")

from projectionSelector import load_projection
from skimage.draw import ellipse
import numpy as np
import matplotlib.pyplot as plt
from tomography import MFI
import matplotlib.style as mplstyle
mplstyle.use('bmh')
plt.rcParams.update({'font.size': 12})
plt.rcParams.update({'figure.dpi': 120})
label_fontsize = 12

# plot = False
plot = True
res = 45
mask_radius = 85
# Load projection matrix ---------------------------------------------------------------------------------

cone_projections_dic = load_projection("complex-view-cone-%d.npy" % res)[0]
cone_projections = cone_projections_dic['projections']
cone_x_grid = cone_projections_dic['x']
cone_y_grid = cone_projections_dic['y']

line_projections_dic = load_projection("line-approximation-%d-etendue.npy" % res)[0]
line_projections = line_projections_dic['projections']
line_x_grid = line_projections_dic['x']
line_y_grid = line_projections_dic['y']

print("Scarcity of LoS: %f" % (len(line_projections.flatten()[line_projections.flatten() > 0.0000001]) / len(line_projections.flatten())))
print("Scarcity of VoS: %f" % (len(cone_projections.flatten()[cone_projections.flatten() > 0.0000001]) / len(cone_projections.flatten())))

# Masks, negative mask: zeros inside vessel, positive mask: zeros outside vessel -------------------------------

n_rows = res
n_cols = res

res_x = 200. / float(n_cols)
res_y = 200. / float(n_rows)  # x,y (mm)

ii, jj = ellipse(n_rows / 2., n_cols / 2., mask_radius / res_y, mask_radius / res_x)
mask_negative = np.ones((n_rows, n_cols), dtype=np.float32)
mask_negative[ii, jj] = 0.
mask_positive = np.zeros((n_rows, n_cols), dtype=np.float32)
mask_positive[ii, jj] = 1.

# Compute the norm of the projections --------------------------------------------------------------------

line_projections *= mask_positive
cone_projections *= mask_positive

for i in range(32):
    print("%d - Line/Cone ratio: " % i, np.sum(line_projections[i]) / np.sum(cone_projections[i]))

# Sensor wise normalization --------------------------------------------

# for i in range(32):
#     line_projections[i] *= np.sum(cone_projections[i]) / np.sum(line_projections[i])

# Matrix wide normalization --------------------------------------------

line_projections *= np.sum(cone_projections) / np.sum(line_projections)

# Plot projections cone and line together ------------------------------------------------------------------------------

cone_plot_x = np.append(cone_x_grid, cone_x_grid[-1] + np.abs(cone_x_grid[1] - cone_x_grid[0])) + \
              0.5 * np.abs(cone_x_grid[1] - cone_x_grid[0])
cone_plot_y = np.append(cone_y_grid, cone_y_grid[-1] - np.abs(cone_y_grid[1] - cone_y_grid[0])) - \
              0.5 * np.abs(cone_y_grid[1] - cone_y_grid[0])

line_plot_x = np.append(line_x_grid, line_x_grid[-1] + np.abs(line_x_grid[1] - line_x_grid[0])) + \
              0.5 * np.abs(line_x_grid[1] - line_x_grid[0])
line_plot_y = np.append(line_y_grid, line_y_grid[-1] - np.abs(line_y_grid[1] - line_y_grid[0])) - \
              0.5 * np.abs(line_y_grid[1] - line_y_grid[0])

lines = [5, 23, 24]
lines = range(0, 16, 4)
cones = []
summed_projections = np.zeros_like(line_projections[0])
for i in lines:
    summed_projections = np.where(summed_projections > line_projections[i]*0.5,
                                  summed_projections,
                                  line_projections[i]*0.5)
for j in cones:
    summed_projections = np.where(summed_projections > cone_projections[j],
                                  summed_projections,
                                  cone_projections[j])
plt.figure(constrained_layout=True)
plt.axis(xmin=0, xmax=85, ymin=-85, ymax=85, option='off')
plt.pcolormesh(cone_x_grid - 0.5 * (cone_x_grid[1] - cone_x_grid[0]),
               cone_y_grid + 0.5 * (cone_y_grid[0] - cone_y_grid[1]),
               summed_projections,
               vmin=None,
               vmax=None)
plt.gca().set_xticks([-50, 0, 50])
plt.gca().set_yticks([-50, 0, 50])
plt.gca().set_aspect('equal', anchor='C')
# plt.pcolormesh(x_grid, y_grid, summed_projections)
# plt.axis('scaled')
plt.axis(xmin=-100, xmax=100, ymin=-100, ymax=100)
circle = plt.Circle((0., 0.), 85., color='w', fill=False)
plt.gca().add_artist(circle)
cbar = plt.colorbar()
cbar.set_label("\nAngle subtended (rad)", fontsize=label_fontsize)
plt.xlabel("R (mm)", fontsize=label_fontsize)
plt.ylabel("z (mm)", fontsize=label_fontsize)
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
