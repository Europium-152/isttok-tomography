"""
Create an emissivity Profile for the experimental phantoms
"""

from __future__ import print_function
from calibrationShots import keys
import numpy as np
from scipy.integrate import dblquad 
import matplotlib.pyplot as plt

# Define an analitical formula for the emissivity. Gaussian emissivity
def em(x,y,mu_x,mu_y,height,fwhm=False,sigma=False):    
    if fwhm: sigma=fwhm/2.3548200450309493    
    return ( 1./ (sigma*np.sqrt(2*np.pi) ) * np.exp( -(x-mu_x)**2/(2*sigma**2) ) )*( 1./ (sigma*np.sqrt(2*np.pi) ) * np.exp( -(y-mu_y)**2/(2*sigma**2) ) )


# Get the positions of the experimental points in xy plane --------------------
xy_positions=[]

for key in keys:
    
    rho   = float(key[0])
    theta = np.radians(float(key[2:]))
    
    x = rho*np.cos(theta)
    y = rho*np.sin(theta)
    xy_positions.append( np.array([x,y]) )
    
    print( '[%.6f,%.6f],' % (x , y) ) 
    

# Discretization Parameters ---------------------------------------------------
 
n_rows = 45  # y-axis number of pixels
n_cols = 45  # x-axis number of pixels

x_min = -100.
x_max = +100.

y_min = -100.
y_max = +100.

# x,y pixel width in mm (resolution)
resx=(x_max-x_min)/n_cols 
resy=(y_max-y_min)/n_rows 

# Create the G emissivity matrix ----------------------------

G=np.zeros((n_cols,n_rows)) # G matrix

index = 0 # Phantom to compute the G profile

# Emissivity parameters ----------------------------------------
mu_x   = xy_positions[index][0] # x coord. of the phantom
mu_y   = xy_positions[index][1] # y coord. of the phantom
fwhm   = 4.                     # Use full width half maximum as the source diameter
height = 1.                     # Height of the emissivity profile

for i in range(n_cols):
    for j in range(n_rows):
        # Four corners of the pixel
        x1 = x_min + i*resx
        x2 = x1 + resx
        y2 = y_max - j*resy
        y1 = y2 - resy
        
        x = (x1+x2)/2.
        y = (y1+y2)/2.
        
        
        if ( (x-mu_x)**2 + (y-mu_y)**2 ) < (5*fwhm)**2:  
            # The value at each pixel is the average value inside (integral divided by area)
            G[i,j],_=dblquad(em,y1,y2,lambda x: x1,lambda x: x2, args=(mu_x,mu_y,height,fwhm),epsrel=0.01)
            G[i,j]/=resx*resy
plt.imshow(G)
plt.colorbar()

