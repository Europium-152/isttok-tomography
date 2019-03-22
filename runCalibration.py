# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 13:13:30 2018

@author: danie
"""
from exportSignals import export_signals
from calibrationShots import shots, keys, times
from callReconstruction import reconstruction
import matplotlib.pyplot as plt
from scipy.ndimage.measurements import center_of_mass
import numpy as np
from scipy.stats import pearsonr as correlation
from scipy.optimize import minimize_scalar

#for key,time in zip(keys,times):
#    
#    # Select shot for calibration and export signals into file
#    shot=shots[key]
#    export_signals(shot)
#    
#    # Perform reconstruction for the signals  
#    (first_g,g_list,x_array,y_array)=reconstruction(time)
#    
#    res=[x_array[0]-x_array[1],y_array[0]-y_array[1]]
#    n_rows=len(y_array)
#    n_cols=len(x_array)


for phantom_number in np.arange(1):
    # key = '3_090'
    # phantom_number = 3
    key = keys[phantom_number]
    time = [times[key]]
    # Select shot for calibration and export signals into file
    shot = shots[key]
    export_signals(key)

    # Load phantom model for comparison
    phantom_model = np.load("phantoms/Phantom-%d.npy" % phantom_number)

    # Perform reconstruction for the signals -------------------------------------------------------------------------------



    # Correlation vs alpha parameter ---------------------------------------------------------------------------------------

    # alphas = np.logspace(-5, -4, 1)
    #
    # corr = []
    # for alpha in alphas:
    #
    #     # Call reconstruction routine --------
    #     (first_g, g_list, x_array_plot, y_array_plot) = reconstruction(time,
    #                                                                    alpha_1=alpha,
    #                                                                    alpha_2=alpha,
    #                                                                    alpha_3=0.0001,
    #                                                                    alpha_4=0)
    #
    #     # Compare with the phantom model -----
    #     corr.append(correlation(g_list[-1].flatten(), phantom_model)[0])
    #
    # plt.figure()
    # plt.plot(alphas, corr)

    # Minimization of alpha ------------------------------------------------------------------------------------------------


    def reconstruction_wrapper(alpha_1):

        # Call reconstruction routine --------
        (first_g, g_list, x_array_plot, y_array_plot) = reconstruction(time,
                                                                       alpha_1=alpha_1,
                                                                       alpha_2=alpha_1,
                                                                       alpha_3=1.,
                                                                       alpha_4=0)

        # Compare with the phantom model -----
        return -correlation(g_list[-1].flatten(), phantom_model)[0]


    # result = minimize_scalar(reconstruction_wrapper, bracket=(0.0001, 0.001), tol=0.01, options={'maxiter': 10})
    # print (result.x)
    #

    #
    # # %% ####################################################################
    # #                                                                       #
    # #                         PLOTTING SPECIFIC                             #
    # #                                                                       #
    # #########################################################################

    # Regularization parameters ----------
    alpha_1 = 0.001
    alpha_2 = alpha_1
    alpha_3 = 1.
    alpha_4 = 0

    (first_g, g_list, x_array_plot, y_array_plot) = reconstruction(time,
                                                                   alpha_1=alpha_1,
                                                                   alpha_2=alpha_2,
                                                                   alpha_3=alpha_3,
                                                                   alpha_4=alpha_4)

    res = [x_array_plot[1] - x_array_plot[0], y_array_plot[0] - y_array_plot[1]]
    n_rows = len(y_array_plot)
    n_cols = len(x_array_plot)

    # plt.figure()
    # plt.axes().set_aspect('equal', 'datalim')
    # plt.pcolormesh(x_array_plot, y_array_plot, first_g, vmin=None, vmax=None)
    # plt.colorbar()
    # circle = plt.Circle((0., 0.), 85., color='w', fill=False)
    # plt.gca().add_artist(circle)

    for G in g_list:

        centroid = center_of_mass(G)
        print ('centroid index : (%.2f, %.2f)' % (centroid[1], centroid[0]))
        center_y = n_rows * res[1]/2. - centroid[0] * res[1]
        center_x = -n_cols * res[0]/2. + centroid[1] * res[0]

        maximum = np.unravel_index(G.argmax(), G.shape)

        max_y = n_rows * res[1]/2. - maximum[0] * res[1] - res[1]/2.
        max_x = -n_cols * res[0]/2. + maximum[1] * res[0] + res[1]/2.

        print ('centroid coords: (%.2f, %.2f)' % (center_x, center_y))

        print('maximum coords: (%.2f, %.2f)' % (max_x, max_y))

        plt.figure()
        plt.axes().set_aspect('equal', 'datalim')
        plt.pcolormesh(x_array_plot, y_array_plot, G, vmin=None, vmax=None)
        plt.title(r"Phantom %d / $\alpha$ = %f" % (phantom_number, alpha_1))
        #plt.imshow(g.reshape((n_rows, n_cols)))
    #    plt.plot(center_x, center_y, 'r+')
    #    plt.plot(max_x, max_y, 'b+')
        plt.colorbar()
        circle = plt.Circle((0., 0.), 85., color='w', fill=False)
        plt.gca().add_artist(circle)
        plt.savefig("phantom-reconstructions/phantom-%d-reconstruction.png" % phantom_number)
        plt.show()
