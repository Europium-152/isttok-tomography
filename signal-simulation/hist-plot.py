import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append("D:/uni/tomography-calibration")
import globalParameters as gp
from calibrationShots import keys
from simulateSignals import simulate_signal


def signal_simulation_histogram(phantom):
    """ Show the super fancy histogram for the simulated versus real signals for a given phantom experiment.

    Parameters
    ----------
    phantom: int or string
        Integer associated with lamp position or string with format "radius_3 digit angle", e.g. "3_045"

    Returns
    -------
    ok: int
        1 if function succeeded
    """
    try:
        phantom_number = keys.index(phantom)
    except ValueError:
        phantom_number = phantom

    fig = plt.figure(figsize=(8, 8))
    vessel = fig.add_axes([0.05, 0.05, 0.4, 0.4])  # [Left, Bottom, Width, Height]
    vessel.set_aspect('equal', anchor='C')
    out = fig.add_axes([0.55, 0.05, 0.4, 0.4])
    top = fig.add_axes([0.05, 0.55, 0.4, 0.4])

    phantom_profile = np.load("../phantoms/Phantom-%d.npy" % phantom_number)
    phantom_profile = phantom_profile.reshape((gp.n_rows, gp.n_cols))

    vessel.pcolormesh(gp.x_array_plot, gp.y_array_plot, phantom_profile)
    vessel.add_artist(plt.Circle((0., 0.), 85., color='w', fill=False))

    f_simulated, f_measured = simulate_signal(phantom_number, plot=False)

    top.bar(np.arange(1, 17), height=f_simulated[:16], width=0.3, align='edge', label='simulated')
    top.bar(np.arange(1, 17) + 0.3, height=f_measured[:16], width=0.3, align='edge', label='real')

    out.barh(np.arange(1, 17), width=f_simulated[16:], height=0.3, align='edge', label='simulated')
    out.barh(np.arange(1, 17) + 0.3, width=f_measured[16:], height=0.3, align='edge', label='real')
    out.invert_yaxis()

    plt.show()

    return 1
