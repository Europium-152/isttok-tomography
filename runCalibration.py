# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 13:13:30 2018

@author: danie
"""

from exportSignals import exportSignals
from calibrationShots import shots,keys,times
from callReconstruction import reconstruction
import matplotlib.pyplot as plt
from scipy.ndimage.measurements import center_of_mass
import numpy as np

#for key,time in zip(keys,times):
#    
#    # Select shot for calibration and export signals into file
#    shot=shots[key]
#    exportSignals(shot)
#    
#    # Perform reconstruction for the signals  
#    (first_g,g_list,x_array,y_array)=reconstruction(time)
#    
#    res=[x_array[0]-x_array[1],y_array[0]-y_array[1]]
#    n_rows=len(y_array)
#    n_cols=len(x_array)

key='3_045'
time=[times[key]]
# Select shot for calibration and export signals into file
shot=shots[key]
exportSignals(key)

# Perform reconstruction for the signals  
(first_g,g_list,x_array,y_array)=reconstruction(time)

res=[x_array[1]-x_array[0],y_array[0]-y_array[1]]
n_rows=len(y_array)
n_cols=len(x_array)

# %% ####################################################################
#                                                                       #
#                         PLOTING SPECIFIC                              #
#                                                                       #
#########################################################################                  

plt.figure()
plt.axes().set_aspect('equal', 'datalim')    
plt.pcolormesh(x_array,y_array,first_g,vmin=0,vmax=0.9)
plt.colorbar()
circle = plt.Circle((0., 0.), 85., color='w', fill=False)
plt.gca().add_artist(circle)    
   
for G in g_list:
 
    centroid=center_of_mass(G)
    print ('centroid index : (%.2f, %.2f)' % (centroid[1],centroid[0]))
    center_y=n_rows*res[1]/2.-centroid[0]*res[1]
    center_x=-n_cols*res[0]/2.+centroid[1]*res[0]
    
    maximum=np.unravel_index(G.argmax(), G.shape)
    
    max_y=n_rows*res[1]/2.-maximum[0]*res[1]-res[1]/2.    
    max_x=-n_cols*res[0]/2.+maximum[1]*res[0]+res[1]/2.
    
    print ('centroid coords: (%.2f, %.2f)' % (center_x,center_y))
    
    print('maximum coords: (%.2f, %.2f)' %(max_x,max_y))
    
    
    plt.figure()
    plt.axes().set_aspect('equal', 'datalim')    
    plt.pcolormesh(x_array,y_array,G,vmin=0,vmax=0.04)
    #plt.imshow(g.reshape((n_rows, n_cols)))
#    plt.plot(center_x, center_y, 'r+')
#    plt.plot(max_x, max_y, 'b+')
    plt.colorbar()
    circle = plt.Circle((0., 0.), 85., color='w', fill=False)
    plt.gca().add_artist(circle) 