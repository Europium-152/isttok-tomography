from projectionSelector import load_projection
from tomography import MFI
from exportSignals import prepare_signals
from calibrationShots import keys, times
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr as correlation


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]


def phantom(phantom_id, plot=True):
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

    try:
        phantom_number = keys.index(phantom_id)
    except ValueError:
        phantom_number = phantom_id
        phantom_id = keys[phantom_number]

    projections = load_projection("complex-view-cone.npy")

    mfi = MFI(projections, width=200., height=200., mask_radius=85.)

    signal_times, signal_data = prepare_signals(phantom_id)

    reconstruction_time = times[phantom_id]
    print("Perform reconstruction for time instant %f" % reconstruction_time)
    time_index, time = find_nearest(signal_times, reconstruction_time)
    print("Associated time index is %d, actual stored time is %f" % (time_index, time))

    # Fixed Regularization constant --------------------------------------------------------------------------

    alpha_1 = 0.01
    alpha_2 = 0.01
    alpha_3 = 1
    alpha_4 = 0

    g_list, first_g = mfi.reconstruction(signals=signal_data[time_index],
                                         stop_criteria=0.,
                                         alpha_1=alpha_1,
                                         alpha_2=alpha_2,
                                         alpha_3=alpha_3,
                                         alpha_4=alpha_4,
                                         max_iterations=20)

    # Adaptive regularization constant -------------------------------------------------------------------

    # alpha_3 = 1
    # alpha_4 = 0
    #
    # phantom_model = np.load("phantoms/Phantom-%d.npy" % phantom_number)
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
    if plot:
        for G in g_list:
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

            plt.figure()
            plt.axes().set_aspect('equal', 'datalim')
            plt.pcolormesh(mfi.x_array_plot, mfi.y_array_plot, G, vmin=None, vmax=None)
            plt.title(r"Phantom %d / $\alpha$ = %f" % (phantom_number, alpha_1))
            # plt.imshow(g.reshape((n_rows, n_cols)))
            # plt.plot(center_x, center_y, 'r+')
            # plt.plot(max_x, max_y, 'b+')
            plt.colorbar()
            circle = plt.Circle((0., 0.), 85., color='w', fill=False)
            plt.gca().add_artist(circle)
            plt.savefig("phantom-reconstructions/phantom-%d-reconstruction.png" % phantom_number)
            plt.show()

    return g_list[0], mfi.x_array_plot, mfi.y_array_plot
