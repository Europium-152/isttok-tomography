

import matplotlib
matplotlib.use("TkAgg")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import MultiLineString, LineString

# -----------------------------------------------------------------------------------------

fname = 'cameras.csv'
print('Reading:', fname)
df = pd.read_csv(fname)

print(df)

# -----------------------------------------------------------------------------------------

n_rows = 80  # y-axis pixel resolution
n_cols = 80  # x-axis pixel resolution

x_min = -100.
x_max = +100.

y_min = -100.
y_max = +100.


def transform(x, y):
    j = int((x-x_min)/(x_max-x_min)*n_cols)
    i = int((y_max-y)/(y_max-y_min)*n_rows)
    return i, j

# -------------------------------------------------------------------------


x_grid = np.linspace(x_min, x_max, num=n_cols+1)
y_grid = np.linspace(y_min, y_max, num=n_rows+1)

grid = []

for x in x_grid:
    grid.append([(x, y_min), (x, y_max)])

for y in y_grid:
    grid.append([(x_min, y), (x_max, y)])

grid = MultiLineString(grid)

# -------------------------------------------------------------------------

projections = []

for row in df.itertuples():
    line = LineString([(row.x0, row.y0), (row.x1, row.y1)])
    projection = np.zeros((n_rows, n_cols))
    for segment in line.difference(grid):
        xx, yy = segment.xy
        x_mean = np.mean(xx)
        y_mean = np.mean(yy)
        (i, j) = transform(x_mean, y_mean)
        projection[i,j] = segment.length
    projections.append(projection)
    
projections = np.array(projections)


print('projections:', projections.shape, projections.dtype)

# -------------------------------------------------------------------------
res_x = (x_max - x_min) / float(n_cols)
res_y = (y_max - y_min) / float(n_rows)

x_coord = np.linspace(x_min + res_x * 0.5, x_max - res_x * 0.5, num=n_cols)
y_coord = np.linspace(y_max - res_y * 0.5, y_min + res_y * 0.5, num=n_rows)

proj_dic = {'x': x_coord, 'y': y_coord, 'projections': projections}

fname = 'projections/line-approximation-80.npy'
print('Writing:', fname)
np.save(fname, [proj_dic])

# -------------------------------------------------------------------------

vmin = 0.
vmax = np.sqrt(((x_max-x_min)/n_cols)**2 + ((y_max-y_min)/n_rows)**2)

ni = 4
nj = 4
figsize = (2*nj, 2*ni)

fig, ax = plt.subplots(ni, nj, figsize=figsize)
for i in range(ni):
    for j in range(nj):
        k = i*nj + j
        ax[i,j].imshow(projections[k], vmin=vmin, vmax=vmax)
        ax[i,j].set_axis_off()

fig.suptitle('projections (top camera)')
plt.savefig('./images/projections-top.png',dpi = 300)
plt.show()

fig, ax = plt.subplots(ni, nj, figsize=figsize)
for i in range(ni):
    for j in range(nj):
        k = i*nj + j + ni*nj
        ax[i,j].imshow(projections[k], vmin=vmin, vmax=vmax)
        ax[i,j].set_axis_off()

fig.suptitle('projections (front camera)')
plt.savefig('./images/projections-front.png',dpi = 300)
plt.show()