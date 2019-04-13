import sys
sys.path.append("../")

from projectionSelector import load_projection
from ProducePhantom import produce_phantom
from tomography import MFI
from exportSignals import prepare_signals
from calibrationShots import keys, times
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr as correlation

plot = True

# Load the phantom ---------------------------------------------------------------------------------------

phantom = produce_phantom(0., 0., 32., 1., 45, plot=True)

# Load projection matrix ---------------------------------------------------------------------------------

projections = load_projection("complex-view-cone.npy")
projections = projections

# Simulate the signal created by the phantom -------------------------------------------------------------

signals = np.dot(projections.reshape((projections.shape[0], -1)), phantom)

# Instantiate reconstruction class -----------------------------------------------------------------------

mfi = MFI(projections, width=200., height=200., mask_radius=85.)

# Run reconstruction -------------------------------------------------------------------------------------

alpha_1 = 5e-8
alpha_2 = 5e-8
alpha_3 = 1
alpha_4 = 0

g_list, first_g = mfi.reconstruction_gpu(signals=signals,
                                     stop_criteria=0,
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
        plt.title(r"$\alpha$ = %f" % alpha_1)
        # plt.imshow(g.reshape((n_rows, n_cols)))
        # plt.plot(center_x, center_y, 'r+')
        # plt.plot(max_x, max_y, 'b+')
        plt.colorbar()
        circle = plt.Circle((0., 0.), 85., color='w', fill=False)
        plt.gca().add_artist(circle)
        # plt.savefig("phantom-reconstructions/phantom-%d-reconstruction.png" % phantom_number)

plt.show()

