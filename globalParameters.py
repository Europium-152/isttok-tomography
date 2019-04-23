import numpy as np

# Discretization Parameters ---------------------------------------------------

n_rows = 80  # y-axis number of pixels
n_cols = 80  # x-axis number of pixels

x_min = -100.
x_max = +100.

y_min = -100.
y_max = +100.

# x,y pixel width in mm (resolution)
res_x = (x_max - x_min) / n_cols
res_y = (y_max - y_min) / n_rows

# x and y arrays for plotting purposes. Coordinates represent the top left corner of each pixel
x_array_plot = (np.arange(n_cols + 1) - n_cols / 2.) * res_x
y_array_plot = (n_rows / 2. - np.arange(n_rows + 1)) * res_y

