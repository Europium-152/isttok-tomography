

import sys

sys.path.append("C:/daniel-tomografia")

from isttok import myLib, baseline
from isttok import signal
from isttok.ISTTOKSignal import ISTTOKSignal
import matplotlib.pyplot as plt
from scipy.ndimage import shift
from scipy.optimize import curve_fit
import numpy as np
from .calibrationShots import shots

plt.close("all")

plt.rcParams.update({'font.size': 18})

keys = [
    # '0_000',
    # '3_000',
    # '3_045',
    # '3_090',
    # '3_135',
    '3_180',
    # '3_225',
    # '3_270',
    # '3_315',
    # '5_000',
    # '5_015',
    # '5_030',
    # '5_045',
    # '5_060',
    # '5_075',
    # '5_090',
    # '5_105',
    # '5_120',
    # '5_135',
    # '5_150',
    # '5_165',
    # '5_180',
    # '5_195',
    # '5_210',
    # '5_225',
    # '5_240',
    # '5_255',
    # '5_270',
    # '5_285',
    # '5_300',
    # '5_315',
    # '5_330',
    # '5_345',
    # '7_000',
    # '7_015',
    # '7_030',
    # '7_045',
    # '7_060',
    # '7_075',
    # '7_090',
    # '7_105',
    # '7_120',
    # '7_135',
    # '7_150',
    # '7_165',
    # '7_180',
    # '7_195',
    # '7_210',
    # '7_225',
    # '7_240',
    # '7_255',
    # '7_270',
    # '7_285',
    # '7_300',
    # '7_315',
    # '7_330',
    # '7_345',
]

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

plot = True
# plt.figure(1, figsize=(20, 12))
signals_data = []
signals_time = []

# shot = shots[shot_id]
# shot = shots['0_000']
time_of_interest = []

for key in keys:
    shot = shots[key]
    signals_data = []
    signals_time = []
    for tag in channels:

        x = ISTTOKSignal(shot, tag=tag, uid=False, time_scale=1e-6, cache_dir='../daniel-tomografia/local/shot-cache/')
        detrend = baseline.baseline(x, deg=1)
        data_raw = x.values - detrend
        time_raw = x.times

        # Moving average and under-sampling using n points
        n = 10
        data = np.cumsum(data_raw, axis=0)
        data = (data[n:] - data[:-n]) / n
        data = data[::n]
        data = np.clip(data, 0., None)
        time = time_raw[n // 2::n]
        time = time[:data.shape[0]]

        signals_data.append(data)
        signals_time.append(time)
        if plot:
            plt.plot(time, data, label=tag)
            if tag == channels[15]:
                plt.title('signals (top camera)')
                plt.xlabel('t (s)')
                plt.legend()
                plt.show()
                plt.figure(2)
            if tag == channels[31]:
                plt.title('signals (front camera)')
                plt.xlabel('t (s)')
                plt.legend()
                plt.show()

    signals_data = np.array(signals_data)
    signals_time = np.array(signals_time)
    i, j = np.unravel_index(np.argmax(signals_data), dims=signals_data.shape)
    max_val = signals_data[i, j]
    while signals_data[i, j] > max_val * 0.5:
        j = j + 1

    time_of_interest.append(signals_time[i, j])

for key, time in zip(keys, time_of_interest):
    print("'" + key + "':", time, ",")

# %%

plt.figure(1)
for tag, time, data in zip(channels, signals_time, signals_data):
    print(time_of_interest, data[time == time_of_interest])
    plt.plot(time_of_interest, data[time == time_of_interest][0], 'ro')
    if tag == channels[15]:
        plt.figure(2)

#
#    # -------------------------------------------------------------------------
#
#    signals_data = np.array(signals_data,dtype=np.float32)
#    signals_time = np.array(signals_time,dtype=np.float32)
#
#    print('signals_data:', signals_data.shape, signals_data.dtype)
#    print('signals_time:', signals_time.shape, signals_time.dtype)
#
#    # -------------------------------------------------------------------------
#
#    fname = 'signals_data.npy'
#    print('Writing:', fname)
#    np.save(fname, signals_data)
#
#    fname = 'signals_time.npy'
#    print('Writing:', fname)
#    np.save(fname, signals_time)
#
#
# if __name__=='__main__':
#    export_signals(sys.argv[1],sys.argv[2])
