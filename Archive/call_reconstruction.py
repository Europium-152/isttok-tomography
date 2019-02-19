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

#time=18000.
#time_index,time=find_nearest(signals_time[0],time)
#f=signals_data[:,time_index]

signals=[]
times=np.arange(18000,19001,500)
for time in times:
    time_index,time=find_nearest(signals_time[0],time)
    signals.append(signals_data[:,time_index])



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

g_list,first_g = mfi(signals=signals,
                projections=projections,
                stop_criteria=stop_criteria,
                alpha_1=alpha_1,
                alpha_2=alpha_2,
                alpha_3=alpha_3,
                max_iterations=max_iterations)

#########################################################################
#                                                                       #
#                         PLOTING SPECIFIC                              #
#                                                                       #
#########################################################################                  

plt.figure()
plt.imshow(first_g.reshape((n_rows, n_cols)))
plt.colorbar()
       
for g in g_list:
 
    g_matrix=g.reshape((n_rows,n_cols))
    centroid=center_of_mass(g_matrix)
    print ('centroid index : (%.2f, %.2f)' % (centroid[1],centroid[0]))
    center_y=n_rows*res[1]/2.-centroid[0]*res[1]
    center_x=-n_cols*res[0]/2.+centroid[1]*res[0]
    
    maximum=unravel_index(g_matrix.argmax(), g_matrix.shape)
    
    max_y=n_rows*res[1]/2.-maximum[0]*res[1]-res[1]/2.    
    max_x=-n_cols*res[0]/2.+maximum[1]*res[0]+res[1]/2.
    
    print ('centroid coords: (%.2f, %.2f)' % (center_x,center_y))
    
    print('maximum coords: (%.2f, %.2f)' %(max_x,max_y))
    
    
    plt.figure()
    plt.axes().set_aspect('equal', 'datalim')
    plt.pcolormesh(x_array_plot,y_array_plot,g.reshape((n_rows, n_cols)))
    #plt.imshow(g.reshape((n_rows, n_cols)))
    plt.plot(center_x, center_y, 'r+')
    plt.plot(max_x, max_y, 'b+')
    plt.colorbar()



















