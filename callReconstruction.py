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



def reconstruction(rec_times):
    """
    Perform a tomographic reconstruction using the current directory files for the designated time instants.
    The algorithm will use each reconstruction as the initial guess for the next one. 
    For this reason you should only provide consecutive time instants.
    
    input:
        rec_times: iterable
            times to use for reconstruction make sure it's coherent with the time axis provided in signals.npy
            
    output:
        first_guess: 2D numpy array
            matrix with the emissivity profile. Last index is the x coordinate
        reconstruction result: 2D numpy array
            matrix with the emissivity profile. Last index is the x coordinate
        x_array: numpy array
            array with x positions according to the provided resolution
        y_array: numpy array
            array with y positions according to the provided resolution
            
            
    """
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
    
#    signals=[]
#    times=np.arange(18000,19001,500)
#    for time in times:
#        time_index,time=find_nearest(signals_time[0],time)
#        signals.append(signals_data[:,time_index])
        
    signals=[]
    for time in rec_times:
        time_index,time=find_nearest(signals_time[0],time)
        signals.append(signals_data[:,time_index])  
    
    
    
    # Reconstruction Resolution -----------------------------------------------
    
    n_rows = projections.shape[1]
    n_cols = projections.shape[2]    
    res=[4.4444,4.4444] # x,y (mm)
    
    # x and y arrays for ploting purposes. Coordinates represent the top left corner of each pixel
    x_array_plot=( np.arange(n_cols+1) - n_cols/2. )*res[0]
    y_array_plot=( n_rows/2. - np.arange(n_rows+1) )*res[1]
    
    # x and y arrays for calculation purposes. Coordinates represent the center of each pixel
    x_array=np.arange(n_cols)*res[0]-n_cols/2.*res[0]
    y_array=n_rows/2.*res[1]-np.arange(n_rows)*res[1]
    
    
    # Convergence parameters --------------------------------------------------
    stop_criteria=1e-4
    max_iterations=10
    
    # Regularization parameters -----------------------------------------------
    alpha_1 = 0.
    alpha_2 = alpha_1
    alpha_3 = alpha_1*10
    
    
    g_list,first_g = mfi(signals=signals,
                    projections=projections,
                    stop_criteria=stop_criteria,
                    alpha_1=alpha_1,
                    alpha_2=alpha_2,
                    alpha_3=alpha_3,
                    max_iterations=max_iterations)
    
    return (first_g,g_list,x_array_plot,y_array_plot)
    




















