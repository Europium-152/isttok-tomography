from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import ellipse
import sys

plt.close("all")

plt.rcParams.update({'font.size': 18})

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return (idx,array[idx])

# -------------------------------------------------------------------------

fname = 'projections.npy'
print('Reading:', fname)
projections = np.load(fname)

print('projections:', projections.shape, projections.dtype)

# -------------------------------------------------------------------------

fname = 'signals_data.npy'
print('Reading:', fname)
signals_data = np.load(fname)

print('signals_data:', signals_data.shape, signals_data.dtype)

fname = 'signals_time.npy'
print('Reading:', fname)
signals_time = np.load(fname)

print('signals_time:', signals_time.shape, signals_time.dtype)

# -------------------------------------------------------------------------

P = projections.reshape((projections.shape[0], -1))

print('P:', P.shape, P.dtype)

# -------------------------------------------------------------------------

n_rows = projections.shape[1]
n_cols = projections.shape[2]

Dh = np.eye(n_rows*n_cols) - np.roll(np.eye(n_rows*n_cols), 1, axis=1)
Dv = np.eye(n_rows*n_cols) - np.roll(np.eye(n_rows*n_cols), n_cols, axis=1)

print('Dh:', Dh.shape, Dh.dtype)
print('Dv:', Dv.shape, Dv.dtype)

# -------------------------------------------------------------------------

ii, jj = ellipse(n_rows//2, n_cols//2, n_rows//2, n_cols//2)
mask = np.ones((n_rows, n_cols))
mask[ii,jj] = 0.

Io = np.eye(n_rows*n_cols) * mask.flatten()

print('Io:', Io.shape, Io.dtype)



# Minimum Fisher Information-----------------------------------------------


time=88000.
time_index,time=find_nearest(signals_time[0],time)
f=signals_data[:,time_index]

alpha_1 = 1e-4
alpha_2 = alpha_1
alpha_3 = alpha_1*10

Pt = np.transpose(P)
PtP = np.dot(Pt, P)
ItIo = np.dot(np.transpose(Io), Io)

# First iteration sets W to 1
W=np.eye(n_rows*n_cols)

DtWDh=np.dot(np.transpose(Dh), np.dot(W, Dh))
DtWDv=np.dot(np.transpose(Dv), np.dot(W, Dv))

inv = np.linalg.inv(PtP + alpha_1*DtWDh + alpha_2*DtWDv + alpha_3*ItIo)
M = np.dot(inv, Pt)
g_old = np.dot(M, f)

plt.figure()
plt.imshow(g_old.reshape((n_rows, n_cols)))
plt.colorbar()

# Then we go into the iterations
i=0
stop_criteria=1e-4
max_iterations=20
while True:
    
    i=i+1;
    
    W=np.diag(1.0/np.abs(g_old))
    
    DtWDh=np.dot(np.transpose(Dh), np.dot(W, Dh))
    DtWDv=np.dot(np.transpose(Dv), np.dot(W, Dv))
    
    inv = np.linalg.inv(PtP + alpha_1*DtWDh + alpha_2*DtWDv + alpha_3*ItIo)
    M = np.dot(inv, Pt)
    g_new = np.dot(M, f)
    
    error=np.sum(np.abs(g_new-g_old))/len(g_new)
    
    print (error)
    
    if error<stop_criteria:
        print ("Minimum Fisher converged after ",i," iterations.")
        break
    
    if i>max_iterations:
        print ("WARNING: Minimum Fisher did not converge after ",i," iterations.")
        break
    
    g_old=np.array(g_new) # Explicitly copy because python will not
        
plt.figure()
plt.imshow(g_new.reshape((n_rows, n_cols)))
plt.colorbar()
    
