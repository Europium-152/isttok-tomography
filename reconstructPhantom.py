from projectionSelector import load_projection
import pandas as pd
# from optimizedC.tomography import MFI
from tomography import MFI
from exportSignals import prepare_signals
from calibrationShots import keys, times
import numpy as np
import matplotlib.pyplot as plt
from signalSimulation.histPlot import magic_histogram
from scipy.stats import pearsonr as correlation
from skimage.measure import compare_ssim
import matplotlib.style as mplstyle
# mplstyle.use('ggplot')
mplstyle.use('bmh')
plt.rcParams.update({'font.size': 12})
plt.rcParams.update({'figure.dpi': 120})

# orange = (251./255., 170./255., 39./255.)
orange = (239./255., 123./255., 7./255.)
blue = (0./255., 159./255., 227./255.)


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]


def rerange(x, lowlim=0., uplim=1.):
    """Remap x onto [min, max] interval"""

    return (x - np.min(x)) / (np.max(x) - np.min(x)) * (uplim - lowlim) + lowlim


script = 'single'  # Choose between 'sweep' or 'single'
df = pd.read_csv('data.csv')

##########################################################################################
                                                                                        ##
#                             parameters for sweep mode                                 ##
                                                                                        ##
if script == 'sweep':                                                                   ##
    res = 45                                                                            ##
    mode = 'c'  # Choose between l or c                                                 ##
    line_interval = [0.00005, 0.001]                                                       ##
    line_pts = 40                                                                       ##
    cone_interval = [0.00005, 0.001]                                                       ##
    cone_pts = 40                                                                       ##
    phantoms = range(21, 22)                                                              ##
    plot_correlation_vs_alpha = True                                                    ##
    plot_best_alpha = True                                                              ##
    plot_all_alphas = False                                                              ##
                                                                                        ##
##########################################################################################


##########################################################################################
                                                                                        ##
#                       parameters for single alpha mode                                ##
                                                                                        ##
elif script == 'single':                                                                ##
    res = 45                                                                            ##
    mode = 'l'  # Choose between l or c                                                 ##
    alpha = 0.000079                                                                  ##
    phantoms = range(21, 22)                                                              ##
    plot = True                                                                         ##
                                                                                        ##
                                                                                        ##
##########################################################################################

else:
    raise ValueError('"%s" is not a valid value for `script`. Choose between "single" and "sweep"')

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

line_projections_dic = load_projection("line-approximation-%d-etendue.npy" % res)[0]
cone_projections_dic = load_projection("complex-view-cone-%d.npy" % res)[0]

line_projections = line_projections_dic['projections'][:32]
cone_projections = cone_projections_dic['projections'][:32]

# Renormalization of geometry matrices --------------------------------------------------------
line_projections *= np.sum(cone_projections.flatten()) / np.sum(line_projections.flatten())

if mode == 'l':
    projections = line_projections
    projections_dic = line_projections_dic

elif mode == 'c':
    projections = cone_projections
    projections_dic = cone_projections_dic

else:
    raise ValueError('Mode must be "l" for line approximation or "c" for cone of view')


mfi = MFI(projections, width=200., height=200., mask_radius=85.)

best_alphas = []
best_correlations = []
best_chisquared = []

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

    if script == 'single':

        alpha_1 = alpha
        alpha_2 = alpha
        alpha_3 = 1
        alpha_4 = 0

        phantom_model = np.load("phantoms-%d-circle/Phantom-%d.npy" % (res, phantom_number))

        reconstruction = mfi.reconstruction_gpu(signals=signal_data[time_index],
                                                stop_criteria=0.02,
                                                alpha_1=alpha_1,
                                                alpha_2=alpha_2,
                                                alpha_3=alpha_3,
                                                alpha_4=alpha_4,
                                                max_iterations=15,
                                                verbose=True)

        g_list = reconstruction._iterations

        print("SSIM: %f" % compare_ssim(rerange(g_list[-1]), rerange(phantom_model.reshape((res, res)))))

        if plot:
            magic_histogram(projections_dic=projections_dic,
                            signals=signal_data[time_index],
                            emissivity=g_list[-1],
                            no_bot=True)

    if script == 'sweep':

        if mode == 'l':
            alphas = np.logspace(np.log10(line_interval[0]), np.log10(line_interval[1]), line_pts)
        elif mode == 'c':
            alphas = np.logspace(np.log10(cone_interval[0]), np.log10(cone_interval[1]), cone_pts)
        else:
            raise ValueError('Mode must be "l" for line approximation or "c" for cone of view')

        alpha_3 = 1
        alpha_4 = 0

        phantom_model = np.load("phantoms-%d-circle/Phantom-%d.npy" % (res, phantom_number))

        correlations = []
        chisquared = []
        converged_list = []

        for alpha_1 in alphas:

            alpha_2 = alpha_1

            reconstruction = mfi.reconstruction_gpu(signals=signal_data[time_index],
                                                    stop_criteria=0.02,
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

            # correlations.append(correlation(g_list[-1].flatten(), phantom_model)[0])
            correlations.append(compare_ssim(rerange(g_list[-1]), rerange(phantom_model.reshape((res, res)))))

            chisquared.append(np.sum((np.dot(mfi._Pt.T, g_list[-1].flatten()) - signal_data[time_index]) ** 2))

            if plot_all_alphas:
                magic_histogram(projections_dic=projections_dic,
                                signals=signal_data[time_index],
                                emissivity=g_list[-1],
                                no_bot=True)

        # Convergence comes in True and False, convert to 1 and 0
        converged_list = np.array([(1 if c else 0) for c in converged_list])
        for i in range(len(converged_list)):
            converged_list[:i] *= converged_list[i]

        # Mask correlation rankings with the list of values that converged
        correlations = np.array(correlations)
        correlations[np.isnan(correlations)] = 0
        masked_correlations = correlations * converged_list

        best_correlations.append(max(masked_correlations))
        best_alphas.append(alphas[np.argmax(masked_correlations)])
        best_chisquared.append(chisquared[np.argmax(masked_correlations)])
        # plt.close('all')

        if plot_correlation_vs_alpha:

            fig, ax1 = plt.subplots()

            lns1 = ax1.semilogx(alphas, correlations, color=orange, label='correlation')
            lns2 = ax1.semilogx(alphas, converged_list * 0.5, label='convergence threshold', linestyle='--')
            ax1.set_xlabel(r'$\lambda$')
            ax1.set_ylabel('correlation', color=orange)
            ax1.set_ylim(-0.05, 1)

            ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

            ax2.set_ylabel(r'$\chi^2$', color=blue)  # we already handled the x-label with ax1
            lns3 = ax2.semilogx(alphas, chisquared, color=blue, label=r'$\chi^2$')
            ax2.set_ylim(-0.5, 10)

            lns = lns1 + lns2 + lns3
            labs = [l.get_label() for l in lns]
            ax2.legend(lns, labs, loc=0)

            if mode == 'c':
                plt.title("Cone")
            elif mode == 'l':
                plt.title("Line")
            else:
                raise ValueError('Mode must be "l" for line approximation or "c" for cone of view')

            plt.tight_layout()

            plt.show()

        if plot_best_alpha:

            reconstruction = mfi.reconstruction_gpu(signals=signal_data[time_index],
                                                    stop_criteria=0.02,
                                                    alpha_1=best_alphas[-1],
                                                    alpha_2=best_alphas[-1],
                                                    alpha_3=alpha_3,
                                                    alpha_4=alpha_4,
                                                    max_iterations=15,
                                                    verbose=True)

            g_list = reconstruction._iterations

            magic_histogram(projections_dic=projections_dic,
                            signals=signal_data[time_index],
                            emissivity=g_list[-1],
                            no_bot=True)
            #
            plt.show()

if script == 'sweep':
    for phantom_id, corr, alpha, chi in zip(phantoms, best_correlations, best_alphas, best_chisquared):
        if mode == 'l':
            df.at[phantom_id, 'corr-los'] = corr
            df.at[phantom_id, 'alpha-los'] = alpha
            df.at[phantom_id, 'chi-los'] = chi
            print("\nResults for phantom %d, using Line of Sight:\n" % phantom_id)
        if mode == 'c':
            df.at[phantom_id, 'corr-vos'] = corr
            df.at[phantom_id, 'alpha-vos'] = alpha
            df.at[phantom_id, 'chi-vos'] = chi
            print("\nResults for phantom %d, using Volume of Sight:\n" % phantom_id)

        print("Best correlation achieved: %f" % corr)
        print("Corresponding alpha: %f" % alpha)
        print("Corresponding chi-s: %f" % chi)

df.to_csv('data.csv', index=False)
df.to_excel('data.xlsx')

# np.save("line_4_correlations.npy", np.array([best_alphas, best_correlations, best_chisquared]))
# Adaptive regularization constant -------------------------------------------------------------------

# alpha_3 = 1
# alpha_4 = 0
#
# phantom_model = np.load("phantoms-45-gaussian/Phantom-%d.npy" % phantom_number)
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
