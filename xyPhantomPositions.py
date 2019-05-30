"""
Create an emissivity Profile for the experimental phantoms-45-gaussian
"""


from calibrationShots import keys
import numpy as np
from scipy.integrate import dblquad
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
mplstyle.use('bmh')
plt.rcParams.update({'font.size': 12})
plt.rcParams.update({'figure.dpi': 120})
label_fontsize = 12


# Define an analytical formula for the emissivity. Gaussian emissivity
def em_gaussian(x, y, mu_x, mu_y, area=1., fwhm=False, sigma=False):
    if fwhm:
        sigma = fwhm / 2.3548200450309493
    return area * \
        (1. / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(x - mu_x) ** 2 / (2 * sigma ** 2))) * \
        (1. / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(y - mu_y) ** 2 / (2 * sigma ** 2)))


def em_circular(x, y, x0, y0, radius, height, outside_value=0):
    """Define a circular shaped emissivity

    Parameters
    ----------
    x, y: float
        Coordinates the input
    x0, y0: float
        Coordinates of the circle center
    radius: float
        Circle radius
    height: float
        Values inside the circle
    outside_value: float, optional
        Values outside the circle and on the boundary. Defaults to 0.

    Returns
    -------
    z: float
        Returns height if point (x, y) lies inside the circle, and returns `outside_value` otherwise
    """

    if ((x - x0) ** 2 + (y - y0) ** 2) < (radius ** 2):
        return height
    else:
        return outside_value


# Get the positions of the experimental points in xy plane --------------------
xy_positions = []

for key in keys:
    rho = 10 * float(key[0])
    theta = np.radians(float(key[2:]))

    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    xy_positions.append(np.array([x, y]))

    # print('[%.6f,%.6f],' % (x, y))

# Discretization Parameters ---------------------------------------------------

n_rows = 45  # y-axis number of pixels
n_cols = 45  # x-axis number of pixels

x_min = -100.
x_max = +100.

y_min = -100.
y_max = +100.

# x,y pixel width in mm (resolution)
res_x = (x_max - x_min) / n_cols
res_y = (y_max - y_min) / n_rows


# Create the G emissivity matrix -----------------------------------------------

G = np.zeros((n_rows, n_cols))  # G matrix

index = 0  # Phantom to compute the G profile

for index in np.arange(0, 57):

    G = np.zeros((n_rows, n_cols))  # G matrix

    # Emissivity parameters ------------------------------------------------------------
    mu_x, mu_y = xy_positions[index]  # phantom coordinates
    fwhm = 4.  # Use full width half maximum as the source diameter (4mm)
    area = 1.  # Value of the gaussian integral

    radius = 2.
    height = 1.

    for i in range(n_rows):
        for j in range(n_cols):
            # Four corners of the pixel
            x1 = x_min + j * res_x
            x2 = x1 + res_x
            y2 = y_max - i * res_y
            y1 = y2 - res_y

            x = (x1 + x2) / 2.
            y = (y1 + y2) / 2.

            # GAUSSIAN PROFILE -------------------------------------------------------------------------------------
            # # Implement a cutoff to calculations where the emissivity is expected to be zero
            # if ((x - mu_x) ** 2 + (y - mu_y) ** 2) < (5 * fwhm) ** 2:
            #     # The value at each pixel is the average value inside (integral divided by area)
            #     G[i, j], _ = dblquad(em_gaussian, y1, y2, lambda _: x1, lambda _: x2, args=(mu_x, mu_y, area, fwhm))
            #     G[i, j] /= res_x * res_y

            # CIRCULAR PROFILE -------------------------------------------------------------------------------------
            if ((x - mu_x) ** 2 + (y - mu_y) ** 2) < (5 * radius) ** 2:
                # The value at each pixel is the average value inside (integral divided by area)
                G[i, j], _ = dblquad(em_circular, y1, y2, lambda _: x1, lambda _: x2, args=(mu_x, mu_y, radius, height))
                G[i, j] /= res_x * res_y

    # # x and y arrays for plotting purposes. Coordinates represent the top left corner of each pixel
    # x_array_plot = (np.arange(n_cols + 1) - n_cols / 2.) * res_x
    # y_array_plot = (n_rows / 2. - np.arange(n_rows + 1)) * res_y
    #
    # plt.figure()
    # plt.pcolormesh(x_array_plot, y_array_plot, G, vmin=None, vmax=None)
    # plt.gca().set_aspect('equal', anchor='C')
    # # plt.axis('scaled')
    # circle = plt.Circle((0., 0.), 85., color='w', fill=False)
    # plt.gca().add_artist(circle)
    # cbar = plt.colorbar()
    # cbar.set_label("\nIntensity (arb. units)", fontsize=label_fontsize)
    # plt.xlabel("R (mm)", fontsize=label_fontsize)
    # plt.ylabel("z (mm)", fontsize=label_fontsize)
    # plt.xlim(-4, 4)
    # plt.ylim(-4, 4)
    # plt.show()

    # Save the phantom emissivity in a file in vector form -------------------------------------------------------------

    np.save("phantoms-45-circle/Phantom-%d.npy" % index, G.flatten())
