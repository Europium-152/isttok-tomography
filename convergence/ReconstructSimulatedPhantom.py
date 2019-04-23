import sys

sys.path.append("../")

from projectionSelector import load_projection
from ProducePhantom import produce_phantom
from tomography import MFI
from exportSignals import prepare_signals
from calibrationShots import keys, times
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.contour as cnt
from scipy.stats import pearsonr as correlation

# plot = False
plot = True

# Load the phantom ---------------------------------------------------------------------------------------

phantom = produce_phantom(mu_x=15., mu_y=15., fwhm=4.5, area=1., resolution=45, plot=False)
# phantom += produce_phantom(-15., 15., 15., 1., 45, plot=False)
# phantom += produce_phantom(5., -15., 15., 1., 45, plot=False)
# phantom += produce_phantom(0., 0., 65., 4., 45, plot=False)

plt.figure()
plt.imshow(phantom.reshape((45, 45)))

# Load projection matrix ---------------------------------------------------------------------------------

projections = load_projection("complex-view-cone-45.npy")

# Simulate the signal created by the phantom -------------------------------------------------------------

signals = np.dot(projections.reshape((projections.shape[0], -1)), phantom)

# Instantiate reconstruction class -----------------------------------------------------------------------

mfi = MFI(projections, width=200., height=200., mask_radius=85.)

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
# for i in range(len(projections)):
for i in [0]:
    summed_projections += projections[i]

plt.figure()
ax = plt.imshow(summed_projections)
cnt.ContourSet(ax, [0.00001])

# %% Run reconstruction -------------------------------------------------------------------------------------

alpha_1 = 10. ** -7.5
alpha_2 = 10. ** -7.5
alpha_3 = 10. ** 0
alpha_4 = 0

# g_list = mfi.reconstruction_gpu(signals=signals,
#                                 stop_criteria=0,
#                                 alpha_1=alpha_1,
#                                 alpha_2=alpha_2,
#                                 alpha_3=alpha_3,
#                                 alpha_4=alpha_4,
#                                 max_iterations=19,
#                                 iterations=True,
#                                 verbose=True,
#                                 guess=None)

# Adaptive regularization constant -------------------------------------------------------------------

# alpha_3 = 1.
# alpha_4 = 0.
#
#
# def comparison(g):
#     return np.sum(np.abs(g.flatten()-phantom))
#
#
# g_list, first_g, alpha_1 = mfi.tomogram(signals=signals,
#                                         stop_criteria=0.,
#                                         comparison=comparison,
#                                         alpha_3=alpha_3,
#                                         alpha_4=alpha_4,
#                                         inner_max_iterations=8,
#                                         outer_max_iterations=8)

# Brute force search for good alpha ---------------------------------------------------------------------
#
# alphas = np.linspace(-9.0, -8.5, 10)
# costs = []
#
#
# def comparison(g):
#     return np.sum(np.abs(g.flatten() - phantom))
#
#
# for alpha in alphas:
#
#     alpha_1 = 10.**alpha
#     alpha_2 = 10.**alpha
#     alpha_3 = 1
#     alpha_4 = 0
#
#     g_list, first_g = mfi.reconstruction_gpu(signals=signals,
#                                              stop_criteria=0,
#                                              alpha_1=alpha_1,
#                                              alpha_2=alpha_2,
#                                              alpha_3=alpha_3,
#                                              alpha_4=alpha_4,
#                                              max_iterations=10,
#                                              verbose=True)
#
#     costs.append(np.log(comparison((g_list[-1]))))
#
# plt.figure()
# plt.plot(alphas, costs)
# # plt.xlim((-9.5, -7))
# plt.show()

#########################################################################
#                                                                       #
#                                PLOTTING                               #
#                                                                       #
#########################################################################

if plot:

    rows = 4
    cols = 5

    fig, axes = plt.subplots(nrows=rows, ncols=cols)
    for ax, G in zip(axes.flat, g_list):
        # centroid = center_of_mass(G)
        # print ('centroid index : (%.2f, %.2f)' % (centroid[1], centroid[0]))
        # center_y = n_rows * res[1]/2. - centroid[0] * res[1]
        # center_x = -n_cols * res[0]/2. + centroid[1] * res[0]
        #
        # maximum = np.unravel_index(G.argmax(), G.shape)
        #
        # max_y = n_rows * res[1]/2. - maximum[0] * res[1] - res[1]/2.
        # max_x = -n_cols * res[0]/2. + maximum[1] * res[0] + res[1]/2.
        #
        # print ('centroid coords: (%.2f, %.2f)' % (center_x, center_y))
        #
        # print('maximum coords: (%.2f, %.2f)' % (max_x, max_y))

        ax.set_aspect('equal', 'datalim')
        im = ax.pcolormesh(mfi.x_array_plot, mfi.y_array_plot, G, vmin=None, vmax=None)
        # ax.title(r"$\alpha$ = %f" % alpha_1)
        circle = plt.Circle((0., 0.), 85., color='w', fill=False)
        ax.add_artist(circle)

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    plt.show()

    # for G in g_list[-2:]:
    #
    #     plt.figure()
    #     plt.axes().set_aspect('equal', 'datalim')
    #     plt.pcolormesh(mfi.x_array_plot, mfi.y_array_plot, G, vmin=None, vmax=None)
    #     plt.title(r"$\alpha$ = %f" % alpha_1)
    #     # plt.imshow(g.reshape((n_rows, n_cols)))
    #     # plt.plot(center_x, center_y, 'r+')
    #     # plt.plot(max_x, max_y, 'b+')
    #     plt.colorbar()
    #     circle = plt.Circle((0., 0.), 85., color='w', fill=False)
    #     plt.gca().add_artist(circle)
    #     # plt.savefig("phantom-reconstructions/phantom-%d-reconstruction.png" % phantom_number)
    #
    # plt.show()
