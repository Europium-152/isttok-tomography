import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
from skimage.draw import ellipse

vessel_radius = 85     # To use as a mask during reconstruction
res_x = 4.444444       # x and y resolution
res_y = 4.444444


def mfi(signals, projections, stop_criteria, alpha_1, alpha_2, alpha_3, alpha_4, max_iterations, guess=None):
    """Apply the minimum fisher reconstruction algorithm for a given set of measurements from tomography.
    mfi is able to perform multiple reconstruction at a time by employing the rolling iteration.
    
    input:
        signals: array or list of arrays
            should be an array of single measurements from each sensor ordered like in "projections", 
            or a list of such arrays
        projections: list of 2D arrays
            list of LoS matrices for each camera ordered like in "signals"
        stop_criteria: float
            average different between iterations to admit convergence as a percentage.
        alpha_1, alpha_2, alpha_3: float
            regularization weights. Horizontal derivative. Vertical derivative. Outside Norm. Inside Norm.
        max_iterations: int
            Maximum number of iterations before algorithm gives up
        guess: Unimplemented
            Yet to be implemented :(
            
    output:
        g_list: array or list of arrays
            Reconstructed g vector, or multiple g vectors if multiple signals were provided
        first_g: array
            First iteration g vector. This is the result of the simple Tikhonov regularization
    """

    # Reconstruction size ----------------------------------------------------------------------------------------------

    n_rows = projections.shape[1]
    n_cols = projections.shape[2]

    # Masks, negative mask: zeros inside vessel, positive mask: zeros outside vessel -----------------------------------

    ii, jj = ellipse(n_rows / 2., n_cols / 2., vessel_radius / res_y, vessel_radius / res_x)
    # ii, jj = ellipse(n_rows / 2., n_cols / 2., n_rows / 2., n_cols / 2.)
    mask_negative = np.ones((n_rows, n_cols))
    mask_negative[ii, jj] = 0.
    mask_positive = np.zeros((n_rows, n_cols))
    mask_positive[ii, jj] = 1.

    # Apply mask to projection matrix and then reshape -----------------------------------------------------------------

    P = (projections*mask_positive).reshape((projections.shape[0], -1))

    # x and y gradient matrices ----------------------------------------------

    Dh = np.eye(n_rows * n_cols) - np.roll(np.eye(n_rows * n_cols), 1, axis=1)
    Dv = np.eye(n_rows * n_cols) - np.roll(np.eye(n_rows * n_cols), n_cols, axis=1)

    print('Dh:', Dh.shape, Dh.dtype)
    print('Dv:', Dv.shape, Dv.dtype)

    # Norm matrix --------------------------------------------------------------

    Io = np.eye(n_rows*n_cols) * mask_negative.flatten()
    Ii = np.eye(n_rows*n_cols) * mask_positive.flatten()
    # Io = np.eye(n_rows * n_cols)

    print('Io:', Io.shape, Io.dtype)

    # Regularization parameters ------------------------------------------------

    # alpha_1 = 10.
    # alpha_2 = alpha_1
    # alpha_3 = 1.

    # stop_criteria = 0.

    # p transpose and PtP ------------------------------------------------------
    Pt = np.transpose(P)
    PtP = np.dot(Pt, P)

    # Norm matrix transposed ---------------------------------------------------
    ItIo = np.dot(np.transpose(Io), Io)
    ItIi = np.dot(np.transpose(Ii), Ii)

    # Discriminate between single and multiple reconstructions mode --------------
    _signals = np.array(signals)
    if len(_signals.shape) == 1:
        f_list = [_signals]
    elif len(_signals.shape) == 2:
        f_list = _signals
    else:
        raise ValueError("signals must be 1 dim for single reconstruction or 2 dim for multiple reconstruction")

    # FIRST ITERATION  -------------------------------------------------------------

    if guess is None:
        # Weight matrix, first iteration sets W to 1 -------------------------------
        W = np.eye(n_rows * n_cols)

        # Fisher information (weighted derivatives) --------------------------------
        DtWDh = np.dot(np.transpose(Dh), np.dot(W, Dh))
        DtWDv = np.dot(np.transpose(Dv), np.dot(W, Dv))

        # Inversion and calculation of vector g, storage of first guess ------------
        inv = np.linalg.inv(PtP + alpha_1 * DtWDh + alpha_2 * DtWDv + alpha_3 * ItIo + alpha_4 * ItIi)
        M = np.dot(inv, Pt)
        g_old = np.dot(M, f_list[0])
        first_g = np.array(g_old)
    else:  # TODO: make sure guess is compatible before implementing!
        print ('Implementation Error: giving an initial guess is not implemented yet, too bad')
        return 0
        g_old = np.array(guess)
        first_g = guess

    g_list = []

    # Iterative process --------------------------------------------------------
    for f in f_list:
        i = 0
        while True:

            i = i + 1

            #            g_old[g_old<1e-20]=1e-20
            W = np.diag(1.0 / np.abs(g_old))

            DtWDh = np.dot(np.transpose(Dh), np.dot(W, Dh))
            DtWDv = np.dot(np.transpose(Dv), np.dot(W, Dv))

            inv = np.linalg.inv(PtP + alpha_1 * DtWDh + alpha_2 * DtWDv + alpha_3 * ItIo + alpha_4 * ItIi)
            M = np.dot(inv, Pt)
            g_new = np.dot(M, f)

            # plt.figure()
            # plt.imshow(g_new.reshape((n_rows, n_cols)))

            # error = np.sum(np.abs((g_new[g_new > 1e-5] - g_old[g_new > 1e-5]) / g_new[g_new > 1e-5])) / len(g_new > 1e-5)
            error = np.sum(np.abs(g_new - g_old)) / np.sum(np.abs(g_new))

            print ("Iteration %d changed by %.4f%%" % (i, error*100.))

            g_old = np.array(g_new)  # Explicitly copy because python will not
            # TODO: Swapping instead of copying

            if error < stop_criteria:
                print ("Minimum Fisher converged after %d iterations." % i)
                break

            if i > max_iterations:
                print ("WARNING: Minimum Fisher did not converge after %d iterations." % i)
                break

        g_list.append(g_new.reshape((n_rows, n_cols)))

    # Return correctly either a single or multiple reconstructions -------------
    if len(_signals.shape) == 1:
        return g_list[0], first_g.reshape((n_rows, n_cols))

    elif len(_signals.shape) == 2:
        return g_list, first_g.reshape((n_rows, n_cols))
