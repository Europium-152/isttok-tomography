from projectionSelector import load_projection
from tomography import MFI
# from tomography import MFI
from exportSignals import prepare_plasma_signals
from calibrationShots import keys, times
import numpy as np
import matplotlib.pyplot as plt
import math
from signalSimulation.histPlot import magic_histogram
from scipy.stats import pearsonr as correlation


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]


def plasma(shot, reconstruction_time, plot=True):
    """ Perform the MFI reconstruction at a given time instant for a certain shot

    Parameters
    ----------
    shot: int
        Shot number e.g. 45765
    reconstruction_time: float
        Time in us (micro-seconds)
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

    # Signal mask, in case there are pixels that dont work ---------------------------------------------------------

    signal_mask = []
    dead_sensors = [24, 26, 34, 37, 38, 41]  # Dead sensors are numbered 0 through 47
    for i in np.arange(48):
        if not (i in dead_sensors):
            signal_mask.append(i)

    projections_dic = load_projection("complex-view-cone-45.npy")[0]
    projections = projections_dic['projections']
    print("Projections:", projections.size)
    signal_times, data = prepare_plasma_signals(shot=shot, plot=True)

    projections = projections[signal_mask]
    signal_data = data[:, signal_mask]

    mfi = MFI(projections, width=200., height=200., mask_radius=85.)

    print(("Perform reconstruction for time instant %f" % reconstruction_time))
    time_index, time = find_nearest(signal_times, reconstruction_time)
    print(("Associated time index is %d, actual stored time is %f" % (time_index, time)))

    # Fixed Regularization constant --------------------------------------------------------------------------

    alpha_1 = 0.01
    alpha_2 = 0.01
    alpha_3 = 1
    alpha_4 = 0

    g_list = mfi.reconstruction_gpu(signals=signal_data[time_index],
                                         stop_criteria=0.10,
                                         alpha_1=alpha_1,
                                         alpha_2=alpha_2,
                                         alpha_3=alpha_3,
                                         alpha_4=alpha_4,
                                         max_iterations=10)

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

    # Magic histogram plotting -----------------------------------------------------------------------------------------
    magic_histogram(projections_dic=projections_dic,
                    signals=data[time_index],
                    emissivity=g_list[-1],
                    dead_signals=dead_sensors)

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
            plt.title((r"Shot #%d / $\alpha$ = %f" + "\n" + r"t = %f $\mu$s") % (shot, alpha_1, time))
            # plt.imshow(g.reshape((n_rows, n_cols)))
            # plt.plot(center_x, center_y, 'r+')
            # plt.plot(max_x, max_y, 'b+')
            plt.colorbar()
            circle = plt.Circle((0., 0.), 85., color='w', fill=False)
            plt.gca().add_artist(circle)
            plt.savefig("D:/desktop/shot45988/%d_%dms%d.png" %
                        (shot, int(math.modf(time*0.001)[1]), int(math.modf(time*0.001)[0]*10)))

    plt.show()
    return g_list[0], mfi.x_array_plot, mfi.y_array_plot


plasma(45988, 310000)
