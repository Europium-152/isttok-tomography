import sys
sys.path.append("D:/uni/daniel-tomografia")

from isttok import myLib,baseline
from isttok import signal 
from isttok.ISTTOKSignal import ISTTOKSignal
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from scipy.ndimage import shift
from scipy.optimize import curve_fit
import numpy as np
from calibrationShots import shots, keys

plt.close("all")

plt.rcParams.update({'font.size': 18})


channels = [
    "cal_top_01",
    "cal_top_02",
    "cal_top_03",
    "cal_top_04",
    "cal_top_05",
    "cal_top_06",
    "cal_top_07",
    "cal_top_08",
    "cal_top_09",
    "cal_top_10",
    "cal_top_11",
    "cal_top_12",
    "cal_top_13",
    "cal_top_14",
    "cal_top_15",
    "cal_top_16",

    "cal_out_01",
    "cal_out_02",
    "cal_out_03",
    "cal_out_04",
    "cal_out_05",
    "cal_out_06",
    "cal_out_07",
    "cal_out_08",
    "cal_out_09",
    "cal_out_10",
    "cal_out_11",
    "cal_out_12",
    "cal_out_13",
    "cal_out_14",
    "cal_out_15",
    "cal_out_16",
        ]


def export_signals(shot_id, plot=False):
    """Prepare tomography signals for reconstruction. Work only with data from the discharge lamp experiments.

    The reconstruction API relies on signals stored in .npy files. The signal preparation pipe-line follows.
    - Load the data from the ISTTOK data base or from the cache folder
    - Apply Signal processing techniques to the loaded data e.g. (detrending, baseline removal, filtering)
    - Save the signals to the target files "signals_time.npy" and "signals_data.npy"
    This files serve as input to the reconstruction algorithm

    Parameters
    ----------
    shot_id: int
        Shot number.
    plot: Bool, optional
        Show plots of the loaded data after processing

    Returns
    -------
    signals_time: list of ND arrays
        List of the time arrays for the data of each tomography sensor
    signals_data: list of ND arrays
        List of value arrays containing the data from each tomography sensor

    """

    signals_data = []
    signals_time = []
    
    shot=shots[shot_id]
    
    if plot: plt.figure(figsize=(20,12))
    
    for tag in channels:
        
        x=ISTTOKSignal(shot,tag=tag,uid=False,time_scale=1e-6,cache_dir='D:/uni/daniel-tomografia/local/shot-cache/')
        detrend=baseline.baseline(x,deg=1)
        data=x.values-detrend
        time=x.times
        
        n = 10
        data = np.cumsum(data, axis=0)
        data = (data[n:]-data[:-n])/n
        data = data[::n]
        data = np.clip(data, 0., None)
        time = time[n//2::n]
        time = time[:data.shape[0]]
        signals_data.append(data)
        signals_time.append(time)
        if plot:
            plt.plot(time, data, label=tag)
            if tag == channels[15]:
                plt.title('signals (top camera)')
                plt.xlabel('t (s)')
                plt.legend()
                plt.figure(figsize=(20,12))
            if tag == channels[31]:
                plt.title('signals (front camera)')
                plt.xlabel('t (s)')
                plt.legend()
            plt.show()
    
            
    # -------------------------------------------------------------------------
    
    signals_data = np.array(signals_data,dtype=np.float32)
    signals_time = np.array(signals_time,dtype=np.float32)
    
    print('signals_data:', signals_data.shape, signals_data.dtype)
    print('signals_time:', signals_time.shape, signals_time.dtype)
    
    # -------------------------------------------------------------------------
    
    fname = 'signals_data.npy'
    print('Writing:', fname)
    np.save(fname, signals_data)
    
    fname = 'signals_time.npy'
    print('Writing:', fname)
    np.save(fname, signals_time)

    return signals_time, signals_data
    

if __name__ == '__main__':
    export_signals(sys.argv[1], True)



