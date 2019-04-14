
import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import ellipse
import sys
from scipy.ndimage.measurements import center_of_mass
from numpy import unravel_index
import scipy
from core import mfi
from scipy.optimize import minimize_scalar
import cupy as cp
import time

plt.close("all")

plt.rcParams.update({'font.size': 18})


class MFI:

    def __init__(self, projections, width=200., height=200., mask_radius=85.):
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

        self._projections = np.array(projections, dtype=np.float32)
        print('projections:', self._projections.shape, self._projections.dtype)

        n_rows = self._projections.shape[1]
        n_cols = self._projections.shape[2]
        res_x = width / float(n_cols)
        res_y = height / float(n_rows)  # x,y (mm)

        # x and y arrays for plotting purposes. Coordinates represent the top left corner of each pixel
        self.x_array_plot = (np.arange(n_cols + 1) - n_cols / 2.) * res_x
        self.y_array_plot = (n_rows / 2. - np.arange(n_rows + 1)) * res_y

        # x and y arrays for calculation purposes. Coordinates represent the center of each pixel
        self.x_array = np.arange(n_cols) * res_x - n_cols / 2. * res_x
        self.y_array = n_rows / 2. * res_y - np.arange(n_rows) * res_y

        # Masks, negative mask: zeros inside vessel, positive mask: zeros outside vessel -------------------------------

        ii, jj = ellipse(n_rows / 2., n_cols / 2., mask_radius / res_y, mask_radius / res_x)
        mask_negative = np.ones((n_rows, n_cols), dtype=np.float32)
        mask_negative[ii, jj] = 0.
        mask_positive = np.zeros((n_rows, n_cols), dtype=np.float32)
        mask_positive[ii, jj] = 1.

        # Apply mask to projection matrix and then reshape -------------------------------------------------------------

        P = (self._projections * mask_positive).reshape((self._projections.shape[0], -1))
        print('P:', P.shape, P.dtype)

        # x and y gradient matrices ----------------------------------------------

        Dh = np.eye(n_rows * n_cols, dtype=np.float32) - np.roll(np.eye(n_rows * n_cols, dtype=np.float32), 1, axis=1)
        Dv = np.eye(n_rows * n_cols, dtype=np.float32) - np.roll(np.eye(n_rows * n_cols, dtype=np.float32), n_cols, axis=1)

        print('Dh:', Dh.shape, Dh.dtype)
        print('Dv:', Dv.shape, Dv.dtype)

        # Norm matrix --------------------------------------------------------------

        Io = np.eye(n_rows * n_cols, dtype=np.float32) * mask_negative.flatten()
        Ii = np.eye(n_rows * n_cols, dtype=np.float32) * mask_positive.flatten()
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

    # @profile
    def reconstruction(self, signals, stop_criteria, alpha_1, alpha_2, alpha_3, alpha_4, max_iterations):
        """Apply the minimum fisher reconstruction algorithm for a given set of measurements from tomography.
        mfi is able to perform multiple reconstruction at a time by employing the rolling iteration.

        input:
            signals: array or list of arrays
                should be an array of single measurements from each sensor ordered like in "projections",
                or a list of such arrays
            stop_criteria: float
                average different between iterations to admit convergence as a percentage.
            alpha_1, alpha_2, alpha_3, alpha_4: float
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
        n_rows = self._n_rows
        n_cols = self._n_cols

        reg = np.array(self._PtP + alpha_3 * self._ItIo + alpha_4 * self._ItIi, dtype=np.float32)

        # Discriminate between single and multiple reconstructions mode --------------
        _signals = np.array(signals, dtype=np.float32)
        if len(_signals.shape) == 1:
            f_list = [_signals]
        elif len(_signals.shape) == 2:
            f_list = _signals
        else:
            raise ValueError("signals must be 1 dim for single reconstruction or 2 dim for multiple reconstruction")

        # -----------------------------  FIRST ITERATION  -------------------------------------------------------------

        # Weight matrix, first iteration sets W to 1 -------------------------------
        W = np.eye(n_rows * n_cols, dtype=np.float32)

        # Fisher information (weighted derivatives) --------------------------------
        DtWDh = np.dot(np.transpose(Dh), np.dot(W, Dh))
        DtWDv = np.dot(np.transpose(Dv), np.dot(W, Dv))

        # Inversion and calculation of vector g, storage of first guess ------------
        inv = np.linalg.inv(reg + alpha_1 * DtWDh + alpha_2 * DtWDv)
        M = np.dot(inv, Pt)
        g_old = np.dot(M, f_list[0])
        first_g = np.array(g_old, dtype=np.float32)

        g_list = []

        # Iterative process --------------------------------------------------------
        for f in f_list:
            for i in range(max_iterations):

                g_old[g_old < 1e-20] = 1e-20
                W = np.diag(1.0 / np.abs(g_old))

                DtWDh = np.dot(np.transpose(Dh), np.dot(W, Dh))
                DtWDv = np.dot(np.transpose(Dv), np.dot(W, Dv))

                inv = np.linalg.inv(reg + alpha_1 * DtWDh + alpha_2 * DtWDv)
                M = np.dot(inv, Pt)
                g_new = np.dot(M, f)

                # plt.figure()
                # plt.imshow(g_new.reshape((n_rows, n_cols)))

                # error = np.sum(np.abs((g_new[g_new > 1e-5] - g_old[g_new > 1e-5]) / g_new[g_new > 1e-5])) / len(g_new > 1e-5)
                error = np.sum(np.abs(g_new - g_old)) / np.sum(np.abs(first_g))

                print("Iteration %d changed by %.4f%%" % (i, error * 100.))

                g_old = np.array(g_new, dtype=np.float32)  # Explicitly copy because python will not

                if error < stop_criteria:
                    print("Minimum Fisher converged after %d iterations." % i)
                    break

                if i > max_iterations:
                    print("WARNING: Minimum Fisher did not converge after %d iterations." % i)
                    break

                g_list.append(g_new.reshape((n_rows, n_cols)))

            return g_list, first_g.reshape((n_rows, n_cols))

    # @profile
    def reconstruction_gpu(self, signals, stop_criteria, alpha_1, alpha_2, alpha_3, alpha_4, max_iterations):
        """Apply the minimum fisher reconstruction algorithm for a given set of measurements from tomography.
        mfi is able to perform multiple reconstruction at a time by employing the rolling iteration.

        input:
            signals: array or list of arrays
                should be an array of single measurements from each sensor ordered like in "projections",
                or a list of such arrays
            stop_criteria: float
                average different between iterations to admit convergence as a percentage.
            alpha_1, alpha_2, alpha_3, alpha_4: float
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
        Dh = cp.array(self._Dh, dtype=cp.float32)
        Dht = cp.transpose(Dh)
        Dv = cp.array(self._Dv, dtype=cp.float32)
        Dvt = cp.transpose(Dv)
        Pt = cp.array(self._Pt, dtype=cp.float32)
        n_rows = self._n_rows
        n_cols = self._n_cols

        reg = cp.array(self._PtP + alpha_3 * self._ItIo + alpha_4 * self._ItIi)

        # Discriminate between single and multiple reconstructions mode --------------
        _signals = cp.array(signals, dtype=cp.float32)
        if len(_signals.shape) == 1:
            f_list = [_signals]
        elif len(_signals.shape) == 2:
            f_list = _signals
        else:
            raise ValueError("signals must be 1 dim for single reconstruction or 2 dim for multiple reconstruction")

        # -----------------------------  FIRST ITERATION  -------------------------------------------------------------

        # First guess to g is uniform plasma distribution --------------------------
        g_old = cp.ones(n_rows * n_cols, dtype=cp.float32)

        # Weight matrix, first iteration sets W to 1 -------------------------------
        W = cp.diag(1.0 / cp.abs(g_old))
        # cp.asnumpy(W)

        # Fisher information (weighted derivatives) --------------------------------
        DtWDh = cp.dot(Dht, cp.dot(W, Dh))
        # cp.asnumpy(DtWDh)
        DtWDv = cp.dot(Dvt, cp.dot(W, Dv))
        # cp.asnumpy(DtWDh)

        # Inversion and calculation of vector g, storage of first guess ------------

        inv = cp.linalg.inv(reg + alpha_1 * DtWDh + alpha_2 * DtWDv)
        # cp.asnumpy(inv)

        M = cp.dot(inv, Pt)
        # cp.asnumpy(M)

        g_old = cp.dot(M, f_list[0])
        # cp.asnumpy(g_old)

        first_g = cp.array(g_old)

        g_list = []

        # Iterative process --------------------------------------------------------
        for f in f_list:
            for i in range(max_iterations):

                g_old[g_old < 1e-20] = 1e-20

                W = cp.diag(1.0 / cp.abs(g_old))
                # cp.asnumpy(W)

                DtWDh = cp.dot(Dht, cp.dot(W, Dh))
                # cp.asnumpy(DtWDh)
                DtWDv = cp.dot(Dvt, cp.dot(W, Dv))
                # cp.asnumpy(DtWDv)

                inv = cp.linalg.inv(reg + alpha_1 * DtWDh + alpha_2 * DtWDv)
                # cp.asnumpy(inv)

                M = cp.dot(inv, Pt)
                # cp.asnumpy(M)

                g_new = cp.dot(M, f)
                # cp.asnumpy(g_new)

                # plt.figure()
                # plt.imshow(g_new.reshape((n_rows, n_cols)))

                # error = np.sum(np.abs((g_new[g_new > 1e-5] - g_old[g_new > 1e-5]) / g_new[g_new > 1e-5])) / len(g_new > 1e-5)
                error = cp.sum(cp.abs(g_new - g_old)) / cp.sum(cp.abs(first_g))
                # cp.asnumpy(error)

                print("Iteration %d changed by %.4f%%" % (i, error * 100.))

                g_old = cp.array(g_new)  # Explicitly copy because python will not
                # cp.asnumpy(g_old)

                if error < stop_criteria:
                    print("Minimum Fisher converged after %d iterations." % i)
                    break

                elif i == (max_iterations - 1):
                    print("WARNING: Minimum Fisher did not converge after %d iterations." % i)
                    break

                g_list.append(cp.asnumpy(g_new.reshape((n_rows, n_cols))))

            return g_list, cp.asnumpy(first_g.reshape((n_rows, n_cols)))

    def tomogram(self, signals, stop_criteria, comparison, alpha_3, alpha_4, inner_max_iterations=10, outer_max_iterations=10):
        """Apply the minimum fisher reconstruction algorithm for a given set of measurements from tomography.
        mfi is able to perform multiple reconstruction at a time by employing the rolling iteration.

        input:
            signals: array or list of arrays
                should be an array of single measurements from each sensor ordered like in "projections",
                or a list of such arrays
            stop_criteria: float
                average different between iterations to admit convergence as a percentage.
            comparison: callable
                Comparison function. This function should take one argument which is a 2D array tomogram,
                and output a scalar that reflects the quality of the fit. This function will be minimized with respect
                to the regularization constant.
                TODO: Implement the chi square option and use it as default
            alpha_3, alpha_4: float
                regularization weights: 3 - Outside Norm. 4 - Inside Norm.
            inner_max_iterations: int
                Maximum number of iterations of the inner MFI loop
            outer_max_iterations: int
                Maximum number of iterations of the outer loop that optimizes the regularization parameter

        output:
            g_list: array or list of arrays
                Reconstructed g vector, or multiple g vectors if multiple signals were provided
            first_g: array
                First iteration g vector. This is the result of the simple Tikhonov regularization
        """

        def reconstruction_wrapper(alpha_iterable):
            # Call reconstruction routine --------
            (g_list, _) = self.reconstruction_gpu(signals,
                                              stop_criteria=stop_criteria,
                                              alpha_1=alpha_iterable,
                                              alpha_2=alpha_iterable,
                                              alpha_3=alpha_3,
                                              alpha_4=alpha_4,
                                              max_iterations=inner_max_iterations)

            # Compare with the phantom model -----
            # return -correlation(g_list[-1].flatten(), phantom_model)[0]
            return comparison(g_list[-1])

        result = minimize_scalar(reconstruction_wrapper,
                                 method='bounded',
                                 bounds=(0., 0.0001),
                                 bracket=(0.000001, 0.0001),
                                 tol=0.0000000001,
                                 options={'maxiter': outer_max_iterations, 'disp': 3})

        g_list, first_g = self.reconstruction_gpu(signals,
                                              stop_criteria=stop_criteria,
                                              alpha_1=result.x,
                                              alpha_2=result.x,
                                              alpha_3=alpha_3,
                                              alpha_4=alpha_4,
                                              max_iterations=inner_max_iterations)

        print("Optimal regularization constant: %f" % result.x)

        return g_list, first_g, result.x
