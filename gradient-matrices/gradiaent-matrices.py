import numpy as np
from scipy.linalg import block_diag

n_cols = 6
n_rows = 4

Dh_element = np.diag(np.ones(n_cols-1), k=1) - np.diag(np.ones(n_cols-1), k=-1)
Dh_element[0, 0] = -2.
Dh_element[0, 1] = 2.
Dh_element[-1, -1] = 2.
Dh_element[-1, -2] = -2.

Dh = block_diag(*(Dh_element for i in range(n_rows)))

Dv_element = np.append(np.diag(2.*np.ones(n_cols)), np.diag(-2.*np.ones(n_cols)), axis=1)
Dv_top = np.append(Dv_element, np.zeros((n_cols, n_cols*(n_rows-2))), axis=1)

Dv_mid = np.append(np.diag(np.ones(n_cols * (n_rows - 2))), np.zeros((n_cols * (n_rows - 2), n_cols * 2)), axis=1) + \
    np.append(np.zeros((n_cols * (n_rows - 2), n_cols * 2)), np.diag(-1.*np.ones(n_cols * (n_rows - 2))), axis=1)

Dv_bot = np.roll(Dv_top, n_cols * (n_rows - 2), axis=1)

Dv = np.append(Dv_top, Dv_mid, axis=0)
Dv = np.append(Dv, Dv_bot, axis=0)

idx = np.arange(n_cols*n_rows).reshape((n_rows, n_cols))

Dh_old = np.eye(n_rows * n_cols, dtype=np.float32) - np.roll(np.eye(n_rows * n_cols, dtype=np.float32), 1, axis=1)
Dv_old = np.eye(n_rows * n_cols, dtype=np.float32) - np.roll(np.eye(n_rows * n_cols, dtype=np.float32), n_cols, axis=1)
