from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import ellipse
import sys
from scipy.ndimage.measurements import center_of_mass
from numpy import unravel_index
import scipy
from core import mfi

plt.close("all")

plt.rcParams.update({'font.size': 18})


class MFI:

    def __init__(self, projections, width=100., height=100., mask_radius=85.):
        """
        Parameters
        ----------
        projections: ND-array 32 x rows x columns
            Projections of each sensor, ordered coherently with "signals", see reconstruction().
            The number of rows and columns in the projection matrices are the number of rows and columns
            in the final reconstructed profile.
        width: float, optional
            With in millimeters of the reconstruction window. The resolution along xx axis will be
            width / number_of_columns
        height: float, optional
            Height in millimeters of the reconstruction window. The resolution along yy axis will be
            height / number_of_rows
        mask_radius: float, optional
            Reconstruction mask. The algorithm will impose a zero emissivity outside the specified radius
        """

        # Reconstruction Size and Resolution -----------------------------------------------

        self._projections = np.array(projections)
        print('projections:', self._projections.shape, self._projections.dtype)

        n_rows = self._projections.shape[1]
        n_cols = self._projections.shape[2]
        res_x = width / n_cols
        res_y = height / n_rows  # x,y (mm)

        # x and y arrays for plotting purposes. Coordinates represent the top left corner of each pixel
        self.x_array_plot = (np.arange(n_cols + 1) - n_cols / 2.) * res_x
        self.y_array_plot = (n_rows / 2. - np.arange(n_rows + 1)) * res_y

        # x and y arrays for calculation purposes. Coordinates represent the center of each pixel
        self.x_array = np.arange(n_cols) * res_x - n_cols / 2. * res_x
        self.y_array = n_rows / 2. * res_y - np.arange(n_rows) * res_y

        # Masks, negative mask: zeros inside vessel, positive mask: zeros outside vessel -------------------------------

        ii, jj = ellipse(n_rows / 2., n_cols / 2., mask_radius / res_y, mask_radius / res_x)
        mask_negative = np.ones((n_rows, n_cols))
        mask_negative[ii, jj] = 0.
        mask_positive = np.zeros((n_rows, n_cols))
        mask_positive[ii, jj] = 1.

        # Apply mask to projection matrix and then reshape -------------------------------------------------------------

        P = (self._projections * mask_positive).reshape((self._projections.shape[0], -1))
        print('P:', P.shape, P.dtype)

        # x and y gradient matrices ----------------------------------------------

        Dh = np.eye(n_rows * n_cols) - np.roll(np.eye(n_rows * n_cols), 1, axis=1)
        Dv = np.eye(n_rows * n_cols) - np.roll(np.eye(n_rows * n_cols), n_cols, axis=1)

        print('Dh:', Dh.shape, Dh.dtype)
        print('Dv:', Dv.shape, Dv.dtype)

        # Norm matrix --------------------------------------------------------------

        Io = np.eye(n_rows * n_cols) * mask_negative.flatten()
        Ii = np.eye(n_rows * n_cols) * mask_positive.flatten()
        # Io = np.eye(n_rows * n_cols)

        print('Io:', Io.shape, Io.dtype)

        # P transpose and PtP ------------------------------------------------------
        Pt = np.transpose(P)
        PtP = np.dot(Pt, P)

        # Norm matrix transposed ---------------------------------------------------
        ItIo = np.dot(np.transpose(Io), Io)
        ItIi = np.dot(np.transpose(Ii), Ii)

        # Aliasing -----------------------------------------------------------------
        self._Dh = Dh
        self._Dv = Dv
        self._Pt = Pt
        self._PtP = PtP
        self._ItIo = ItIo
        self._ItIi = ItIi
        self._n_rows = n_rows
        self._n_cols = n_cols

    def reconstruction(self, signals, stop_criteria, alpha_1, alpha_2, alpha_3, alpha_4, max_iterations):
        """Apply the minimum fisher reconstruction algorithm for a given set of measurements from tomography.
        mfi is able to perform multiple reconstruction at a time by employing the rolling iteration.

        input:
            signals: array or list of arrays
                should be an array of single measurements from each sensor ordered like in "projections",
                or a list of such arrays
            stop_criteria: float
                average different between iterations to admit convergence as a percentage.
            alpha_1, alpha_2, alpha_3: float
                regularization weights. Horizontal derivative. Vertical derivative. Outside Norm. Inside Norm.
            max_iterations: int
                Maximum number of iterations before algorithm gives up

        output:
            g_list: array or list of arrays
                Reconstructed g vector, or multiple g vectors if multiple signals were provided
            first_g: array
                First iteration g vector. This is the result of the simple Tikhonov regularization
        """

        # Aliasing for cleaner code --------------------------------------------------
        Dh = self._Dh
        Dv = self._Dv
        Pt = self._Pt
        PtP = self._PtP
        ItIo = self._ItIo
        ItIi = self._ItIi
        n_rows = self._n_rows
        n_cols = self._n_cols

        # Discriminate between single and multiple reconstructions mode --------------
        _signals = np.array(signals)
        if len(_signals.shape) == 1:
            f_list = [_signals]
        elif len(_signals.shape) == 2:
            f_list = _signals
        else:
            raise ValueError("signals must be 1 dim for single reconstruction or 2 dim for multiple reconstruction")

        # -----------------------------  FIRST ITERATION  -------------------------------------------------------------

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

                print("Iteration %d changed by %.4f%%" % (i, error * 100.))

                g_old = np.array(g_new)  # Explicitly copy because python will not
                # TODO: Swapping instead of copying

                if error < stop_criteria:
                    print("Minimum Fisher converged after %d iterations." % i)
                    break

                if i > max_iterations:
                    print("WARNING: Minimum Fisher did not converge after %d iterations." % i)
                    break

            g_list.append(g_new.reshape((n_rows, n_cols)))

        # Return correctly either a single or multiple reconstructions -------------
        if len(_signals.shape) == 1:
            return g_list[0], first_g.reshape((n_rows, n_cols))

        elif len(_signals.shape) == 2:
            return g_list, first_g.reshape((n_rows, n_cols))



