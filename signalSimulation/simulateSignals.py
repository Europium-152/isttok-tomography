import numpy as np
from exportSignals import export_signals
import matplotlib.pyplot as plt
from calibrationShots import shots, keys, times
import os
_module_path = os.path.dirname(__file__)

# plt.close("all")
#
# plt.rcParams.update({'font.size': 18})


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]


def simulate_signal(phantom_number, projections, reconstruction_time='auto', plot=True):
    """ For a given phantom number return the simulated data with the current projection method and the real signals.
    The simulated signal is normalized to have the same total intensity as the actual measured signal

    Parameters:
        phantom_number: int
            The number associated with the lamp position
        projections: nd array
            Projections to use for the simulated signal.
        reconstruction_time: float, optional
            Time instant to perform reconstruction. If set to 'auto', the time instant will be retrieved from the list
        plot: Bool, optional
            Show the plotted signals. Defaults to True
    Returns:
        f_simulated: 1x32 ND-array
            Simulated signals for the 32 sensors
        f_measured: 1x32 ND-array
            Actually measured signals during the experiments

    """
    # Projections vector p ------------------------------------------------------

    # fname = '../projections.npy'
    # print('Reading:', fname)
    # projections = np.load(fname)

    print('projections:', projections.shape, projections.dtype)

    P = projections.reshape((projections.shape[0], -1))

    print('P:', P.shape, P.dtype)

    # Emissivity vector g of the simulated phantom ---------------------------------------------------------------------

    g = np.load(os.path.join(_module_path, "../phantoms-45-gaussian/Phantom-%d.npy") % phantom_number)

    # Obtain the simulated signals, simulated f vector -----------------------------------------------------------------

    f_simulated = np.dot(P, g)

    # Obtain the actual signals from phantom measurements and vector f -------------------------------------------------

    key = keys[phantom_number]
    signals_time, signals_data = export_signals(key)

    print('signals_data:', signals_data.shape, signals_data.dtype)
    print('signals_time:', signals_time.shape, signals_time.dtype)

    if reconstruction_time == 'auto':
        time = times[keys[phantom_number]]
        time_index, time = find_nearest(signals_time[0], time)
        f_measured = signals_data[:, time_index]
    else:
        time = reconstruction_time
        time_index, time = find_nearest(signals_time[0], time)
        f_measured = signals_data[:, time_index]

    # Normalize simulated f vector -------------------------------------------------------------------------------------

    # CHOOSE THIS Camera-wise normalization -----------------------------
    # f_simulated[:16] *= np.sum(f_measured[:16])/np.sum(f_simulated[:16])
    # f_simulated[16:] *= np.sum(f_measured[16:])/np.sum(f_simulated[16:])

    # OR THIS Global normalization, avoids reflections ---------------------
    f_simulated *= \
        np.sum(f_measured[f_simulated > (np.sum(f_simulated) / 32000.)]) / \
        np.sum(f_simulated[f_simulated > (np.sum(f_simulated) / 32000.)])

    # Plotting measured and simulated signals --------------------------------------------------------------------------

    if plot:
        bar_width = 0.3
        plt.figure()
        plt.bar(np.arange(16), f_simulated[:16], bar_width, align='edge')
        plt.bar(np.arange(16) + bar_width, f_measured[:16], bar_width, align='edge')
        plt.figure()
        plt.bar(np.arange(16, 32), f_simulated[16:], bar_width, align='edge')
        plt.bar(np.arange(16, 32) + bar_width, f_measured[16:], bar_width, align='edge')
        plt.show()

    return f_simulated, f_measured
