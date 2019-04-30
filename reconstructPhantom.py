from projectionSelector import load_projection
# from optimizedC.tomography import MFI
from tomography import MFI
from exportSignals import prepare_signals
from calibrationShots import keys, times
import numpy as np
import matplotlib.pyplot as plt
from signalSimulation.histPlot import magic_histogram
from scipy.stats import pearsonr as correlation


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]


plot = True
# def phantom(phantom_id, plot=True):
""" Perform the MFI reconstruction for the phantom identified by 'phantom_id'

Parameters
----------
phantom_id: int or string
    Integer associated with lamp position or string with format "radius_3 digit angle", e.g. "3_045"
plot: Boolean, optional
    Whether or not to plot the resulting reconstruction. Defaults to True

Returns
-------
emissivity_matrix: 2D array
    Matrix of the reconstructed emissivity. x is the last index as usual
x_array_plot: 1D array
    X coordinates of the left sides of each pixel in the reconstruction
y_array_plot: 1D array
    Y coordinates of the top sides of each pixel in the reconstruction

"""

# alphas = np.logspace(np.log10(0.06), np.log10(100), 15)
alphas = np.logspace(np.log10(0.0003), np.log10(0.0015), 15)
#
# projections_dic = load_projection("line-approximation-45.npy")[0]
projections_dic = load_projection("complex-view-cone-45.npy")[0]

projections = projections_dic['projections']
projections = projections[:32]

mfi = MFI(projections, width=200., height=200., mask_radius=85.)

# Enable line profiling for performance analyses ------------------------------------------

best_alphas = []
best_correlations = []
best_chisquared = []
phantoms = range(57)

for phantom_id in phantoms:

    try:
        phantom_number = keys.index(phantom_id)
    except ValueError:
        phantom_number = phantom_id
        phantom_id = keys[phantom_number]

    signal_times, signal_data = prepare_signals(phantom_id)

    reconstruction_time = times[phantom_id]
    print(("Perform reconstruction for time instant %f" % reconstruction_time))
    time_index, time = find_nearest(signal_times, reconstruction_time)
    print(("Associated time index is %d, actual stored time is %f" % (time_index, time)))

    # Fixed Regularization constant --------------------------------------------------------------------------

    # alpha_1 = 0.002
    # alpha_2 = 0.002
    # alpha_3 = 1
    # alpha_4 = 0
    #
    # g_list = mfi.reconstruction_gpu(signals=signal_data[time_index],
    #                                 stop_criteria=0.01,
    #                                 alpha_1=alpha_1,
    #                                 alpha_2=alpha_2,
    #                                 alpha_3=alpha_3,
    #                                 alpha_4=alpha_4,
    #                                 max_iterations=15,
    #                                 verbose=True)

    # Manual search for regularization constant ------------------------------------------------------------


    alpha_3 = 1
    alpha_4 = 0

    phantom_model = np.load("phantoms-45/Phantom-%d.npy" % phantom_number)

    correlations = []
    chisquared = []
    converged_list = []

    for alpha_1 in alphas:

        alpha_2 = alpha_1

        reconstruction = mfi.reconstruction_gpu(signals=signal_data[time_index],
                                        stop_criteria=0.05,
                                        alpha_1=alpha_1,
                                        alpha_2=alpha_2,
                                        alpha_3=alpha_3,
                                        alpha_4=alpha_4,
                                        max_iterations=15,
                                        verbose=True)

        g_list = reconstruction._iterations

        converged_list.append(reconstruction.converged())

        # # Flexible correlation --------------------------------------------------
        # flexible_correlations = []
        # for i in range(-2, 3):
        #     for j in range(-2, 3):
        #

        correlations.append(correlation(g_list[-1].flatten(), phantom_model)[0])

        chisquared.append(np.sum((np.dot(mfi._Pt.T, g_list[-1].flatten()) - signal_data[time_index]) ** 2))

    # Convergence comes in True and False, convert to 1 and 0
    converged_list = np.array([(1 if c else 0) for c in converged_list])
    for i in range(len(converged_list)):
        converged_list[:i] *= converged_list[i]

    # Mask correlation rankings with the list of values that converged
    masked_correlations = np.array(correlations) * converged_list

    best_correlations.append(max(masked_correlations))
    best_alphas.append(alphas[np.argmax(masked_correlations)])
    best_chisquared.append(chisquared[np.argmax(masked_correlations)])
    # plt.close('all')

    # fig, ax1 = plt.subplots()
    #
    # color = 'tab:red'
    # ax1.semilogx(alphas, correlations, color=color)
    # ax1.semilogx(alphas, converged_list * 0.5, linestyle='--')
    # ax1.set_xlabel('alpha')
    # ax1.set_ylabel('correlation', color=color)
    # ax1.set_ylim(0, 1)
    #
    # ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    #
    # color = 'tab:blue'
    # ax2.set_ylabel(r'$\chi^2$', color=color)  # we already handled the x-label with ax1
    # ax2.semilogx(alphas, chisquared, color=color)
    # ax2.set_ylim(0, 10)
    #
    # reconstruction = mfi.reconstruction_gpu(signals=signal_data[time_index],
    #                                         stop_criteria=0.05,
    #                                         alpha_1=best_alphas[-1],
    #                                         alpha_2=best_alphas[-1],
    #                                         alpha_3=alpha_3,
    #                                         alpha_4=alpha_4,
    #                                         max_iterations=15,
    #                                         verbose=True)
    #
    # g_list = reconstruction._iterations
    #
    # magic_histogram(projections_dic=projections_dic,
    #                 signals=signal_data[time_index],
    #                 emissivity=g_list[-1],
    #                 no_bot=True)
    #
    # plt.show()

np.save("cone_3_correlations.npy", np.array([best_alphas, best_correlations, best_chisquared]))
# Adaptive regularization constant -------------------------------------------------------------------

# alpha_3 = 1
# alpha_4 = 0
#
# phantom_model = np.load("phantoms-45/Phantom-%d.npy" % phantom_number)
#
# def comparison(G):
#     return -correlation(G.flatten(), phantom_model)[0]
#
# g_list, first_g, alpha_1 = mfi.tomogram(signals=signal_data[time_index],
#                                         stop_criteria=0.,
#                                         comparison=comparison,
#                                         alpha_3=alpha_3,
#                                         alpha_4=alpha_4,
#                                         max_iterations=4)

#########################################################################
#                                                                       #
#                                PLOTTING                               #
#                                                                       #
#########################################################################

# Magic histogram plotting -----------------------------------------------------------------------------------------
# magic_histogram(projections_dic=projections_dic,
#                 signals=signal_data[time_index],
#                 emissivity=g_list[-1],
#                 no_bot=True)

# if plot:
#     for G in g_list:
#         # centroid = center_of_mass(G)
#         # print ('centroid index : (%.2f, %.2f)' % (centroid[1], centroid[0]))
#         # center_y = n_rows * res[1]/2. - centroid[0] * res[1]
#         # center_x = -n_cols * res[0]/2. + centroid[1] * res[0]
#         #
#         # maximum = np.unravel_index(G.argmax(), G.shape)
#         #
#         # max_y = n_rows * res[1]/2. - maximum[0] * res[1] - res[1]/2.
#         # max_x = -n_cols * res[0]/2. + maximum[1] * res[0] + res[1]/2.
#         #
#         # print ('centroid coords: (%.2f, %.2f)' % (center_x, center_y))
#         #
#         # print('maximum coords: (%.2f, %.2f)' % (max_x, max_y))
#
#         plt.figure()
#         plt.axes().set_aspect('equal', 'datalim')
#         plt.pcolormesh(mfi.x_array_plot, mfi.y_array_plot, G, vmin=None, vmax=None)
#         plt.title(r"Phantom %d / $\alpha$ = %f" % (phantom_number, alpha_1))
#         # plt.imshow(g.reshape((n_rows, n_cols)))
#         # plt.plot(center_x, center_y, 'r+')
#         # plt.plot(max_x, max_y, 'b+')
#         plt.colorbar()
#         circle = plt.Circle((0., 0.), 85., color='w', fill=False)
#         plt.gca().add_artist(circle)
#         # plt.savefig("phantom-reconstructions/phantom-%d-reconstruction.png" % phantom_number)
#
# plt.show()
#     # return g_list[0], np.array(mfi.x_array_plot), np.array(mfi.y_array_plot)
