

import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import ellipse
import sys
from scipy.ndimage.measurements import center_of_mass
from numpy import unravel_index
import scipy

plt.close("all")

plt.rcParams.update({'font.size': 18})

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return (idx,array[idx])

#########################################################################
#                                                                       #
#                    PREPARATION SPECIFIC                               #
#                                                                       #
#########################################################################
    
# Projections vector p ------------------------------------------------------

fname = 'projections.npy'
print('Reading:', fname)
projections = np.load(fname)

print('projections:', projections.shape, projections.dtype)

P = projections.reshape((projections.shape[0], -1))

print('P:', P.shape, P.dtype)

# Signals and vector f -----------------------------------------------------

fname = 'signals_data.npy'
print('Reading:', fname)
signals_data = np.load(fname)

print('signals_data:', signals_data.shape, signals_data.dtype)

fname = 'signals_time.npy'
print('Reading:', fname)
signals_time = np.load(fname)

print('signals_time:', signals_time.shape, signals_time.dtype)

time=18000.
time_index,time=find_nearest(signals_time[0],time)
f=signals_data[:,time_index]


# Reconstruction Resolution -----------------------------------------------

n_rows = projections.shape[1]
n_cols = projections.shape[2]    
res=[4.4444,4.4444] # x,y (mm)

# x and y arrays for ploting purposes
x_array_plot=( np.arange(n_cols+1) - n_cols/2. )*res[0]
y_array_plot=( n_rows/2. - np.arange(n_rows+1) )*res[1]

# x and y arrays for calculation purposes
x_array=np.arange(n_cols)*res[0]-n_cols/2.*res[0]
y_array=n_rows/2.*res[1]-np.arange(n_rows)*res[1]


# Convergence parameters --------------------------------------------------
stop_criteria=1e-4
max_iterations=10

# Regularization parameters -----------------------------------------------
alpha_1 = 1e-5
alpha_2 = alpha_1
alpha_3 = alpha_1*10

#########################################################################
#                                                                       #
#                    RECONSTRUCTION SPECIFIC                            #
#                                                                       #
#########################################################################

# x and y grandient matrices ----------------------------------------------

Dh = np.eye(n_rows*n_cols) - np.roll(np.eye(n_rows*n_cols), 1, axis=1)
Dv = np.eye(n_rows*n_cols) - np.roll(np.eye(n_rows*n_cols), n_cols, axis=1)

print('Dh:', Dh.shape, Dh.dtype)
print('Dv:', Dv.shape, Dv.dtype)

# norm matrix --------------------------------------------------------------

ii, jj = ellipse(n_rows//2, n_cols//2, n_rows//2, n_cols//2)
mask = np.ones((n_rows, n_cols))
mask[ii,jj] = 0.

Io = np.eye(n_rows*n_cols) * mask.flatten()

print('Io:', Io.shape, Io.dtype)


# p transpose and PtP ------------------------------------------------------
Pt = np.transpose(P)
PtP = np.dot(Pt, P)

# Norm matrix transposed ---------------------------------------------------
ItIo = np.dot(np.transpose(Io), Io)


######################  FIRST ITERATION   ##################################

# Weight matrix, first iteration sets W to 1 -------------------------------
W=np.eye(n_rows*n_cols)

# Fisher information (weighted derivatives) --------------------------------
DtWDh=np.dot(np.transpose(Dh), np.dot(W, Dh))
DtWDv=np.dot(np.transpose(Dv), np.dot(W, Dv))

# Inversion and calculation of vector g, storage of first guess ------------
inv = np.linalg.inv(PtP + alpha_1*DtWDh + alpha_2*DtWDv + alpha_3*ItIo)
M = np.dot(inv, Pt)
g_old = np.dot(M, f)
first_g = np.array(g_old)

# Iterative process --------------------------------------------------------
i=0
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
                          # TODO: Swaping instead of copying
                          
#########################################################################
#                                                                       #
#                         PLOTING SPECIFIC                              #
#                                                                       #
#########################################################################                  

plt.figure()
plt.imshow(first_g.reshape((n_rows, n_cols)))
plt.colorbar()
        
g_matrix=g_new.reshape((n_rows,n_cols))
centroid=center_of_mass(g_matrix)
print ('centroid in index coordinates:',(centroid[1],centroid[0]))
center_y=n_rows*res[1]/2.-centroid[0]*res[1]
center_x=-n_cols*res[0]/2.+centroid[1]*res[0]

maximum=unravel_index(g_matrix.argmax(), g_matrix.shape)

max_y=n_rows*res[1]/2.-maximum[0]*res[1]-res[1]/2.    
max_x=-n_cols*res[0]/2.+maximum[1]*res[0]+res[1]/2.

print('coordinates of maximum:',(max_x,max_y))

print ('centroid in space coordinates:', (center_x,center_y))
plt.figure()
plt.axes().set_aspect('equal', 'datalim')
plt.pcolormesh(x_array_plot,y_array_plot,g_new.reshape((n_rows, n_cols)))
#plt.imshow(g_new.reshape((n_rows, n_cols)))
plt.plot(center_x, center_y, 'r+')
plt.plot(max_x, max_y, 'b+')
plt.colorbar()

    
