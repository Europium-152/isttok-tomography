# ISTTOK Tokamak Engineering and Operation

This repository contains the source code for the tomography course at the ISTTOK training program on [Tokamak Engineering and Operation](https://isttok.tecnico.ulisboa.pt/~isttok.daemon/index.php?title=Training).

The code is based on the work from the original repository but for a different geometry with solely 2 cameras and different detectors

## Cameras

- Run `cameras.py` to find the lines of sight for each camera.

    - The three cameras are referred to as top (t), front (f) and bottom (b).

    - The pinhole and detector positions are provided in the code.
    
    - The vessel has a circular cross section that is assumed to be centered at (0,0) in the xy-plane.
    
    - An output file `cameras.csv` will be created with the start and end positions for each line of sight.
    
![cameras](https://raw.githubusercontent.com/diogodcarvalho/isttok-tomography/master/images/cameras.png)

## Projections

- Run `projections.py` to find the projections from pixel values to detector measurements.

    - The pixel resolution for the x- and y-axis are defined in the code.
    
    - When a line of sight crosses a pixel, the contribution of that pixel is assumed to be proportionate to the length of the intersection between the line and the pixel.
    
    - When a line of sight does not cross a pixel, the contribution of that pixel is zero, since there is no intersection.

    - The projections will be saved to the output file `projections.npy`.

![projections-top](https://raw.githubusercontent.com/diogodcarvalho/isttok-tomography/master/images/projections-top.png)
![projections-front](https://raw.githubusercontent.com/diogodcarvalho/isttok-tomography/master/images/projections-front.png)

## Signals

- Run `signals.py` to get the signals from each camera, for a given shot number.

    - The code uses the SDAS API that can be downloaded and installed from [here](http://metis.ipfn.ist.utl.pt/CODAC/IPFN_Software/SDAS/Access/Python).
    
    - The shot number is indicated in the code.
    
    - The data acquisition channels that correspond to each camera detector are indicated in the code.
    
    - The signal offset is removed based on the signal average for _t_ < 0 s.
    
    - The signals are subsampled from 2 MHz to 10 kHz.
    
    - The signals are clipped to zero to remove any negative values.
    
    - The signals data and time are stored separately in `signals_data.npy` and `signals_time.npy`.
    
![signals-top](https://raw.githubusercontent.com/diogodcarvalho/isttok-tomography/master/images/signals-top.png)
![signals-front](https://raw.githubusercontent.com/diogodcarvalho/isttok-tomography/master/images/signals-front.png)

## Reconstructions

- Run `reconstructions.py` to reconstruct the plasma profile at specific time points.

    - The regularization is based on the horizontal and vertical differences between pixels.
    
    - The pixels outside the vessel have an additional regularization imposed on them.

    - The regularization weights are indicated in the code.
    
    - The time points for the reconstructions are indicated in the code.
    
![reconstructions](https://raw.githubusercontent.com/diogoff/isttok-tomography/master/images/reconstructions.png)
