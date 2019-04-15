import numpy as np
from scipy.integrate import dblquad
import matplotlib.pyplot as plt


# Emissivity Function ---------------------------------------------------------
def em(x, y, mu_x, mu_y, area=1., fwhm=False, sigma=False):
    if fwhm:
        sigma = fwhm / 2.3548200450309493
    return area * \
           (1. / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(x - mu_x) ** 2 / (2 * sigma ** 2))) * \
           (1. / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(y - mu_y) ** 2 / (2 * sigma ** 2)))


def produce_phantom(mu_x, mu_y, fwhm, area, resolution, plot=True):

    # Discretization Parameters ---------------------------------------------------

    n_rows = resolution  # y-axis number of pixels
    n_cols = resolution  # x-axis number of pixels

    x_min = -100.
    x_max = +100.

    y_min = -100.
    y_max = +100.

    # x,y pixel width in mm (resolution)
    res_x = (x_max - x_min) / n_cols
    res_y = (y_max - y_min) / n_rows

    # Create the G emissivity matrix -----------------------------------------------

    G = np.zeros((n_rows, n_cols))  # G matrix

    # Emissivity parameters ------------------------------------------------------------
    # mu_x, mu_y = (0, 0)  # phantom coordinates
    # fwhm = 4.  # Use full width half maximum as the source diameter (4mm)
    # area = 1.  # Value of the gaussian integral

    for i in range(n_rows):
        for j in range(n_cols):
            # Four corners of the pixel
            x1 = x_min + j * res_x
            x2 = x1 + res_x
            y2 = y_max - i * res_y
            y1 = y2 - res_y

            x = (x1 + x2) / 2.
            y = (y1 + y2) / 2.

            if ((x - mu_x) ** 2 + (y - mu_y) ** 2) < (5 * fwhm) ** 2:
                # The value at each pixel is the average value inside (integral divided by area)
                G[i, j], _ = dblquad(em, y1, y2, lambda _: x1, lambda _: x2, args=(mu_x, mu_y, area, fwhm))
                G[i, j] /= res_x * res_y

    # x and y arrays for plotting purposes. Coordinates represent the top left corner of each pixel
    x_array_plot = (np.arange(n_cols + 1) - n_cols / 2.) * res_x
    y_array_plot = (n_rows / 2. - np.arange(n_rows + 1)) * res_y

    if plot:
        plt.figure()
        plt.pcolormesh(x_array_plot, y_array_plot, G, vmin=None, vmax=None)
        plt.colorbar()
        plt.show()

    return G.flatten()
