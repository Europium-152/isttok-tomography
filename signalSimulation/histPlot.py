import matplotlib.pyplot as plt
import numpy as np
import sys
import os
_module_path = os.path.dirname(__file__)
sys.path.append("D:/uni/tomography-calibration")
from projectionSelector import load_projection
import globalParameters as gp
from calibrationShots import keys
from .simulateSignals import simulate_signal
from skimage.draw import ellipse
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
import matplotlib.style as mplstyle
# mplstyle.use('ggplot')
mplstyle.use('bmh')
plt.rcParams.update({'font.size': 12})
plt.rcParams.update({'figure.dpi': 120})

majorLocator = MultipleLocator(5)  # Step of the major ticks
majorFormatter = FormatStrFormatter('%d')  # String format for major ticks
minorLocator = MultipleLocator(1)  # Step of the minor ticks


def signal_simulation_histogram(phantom):
    """ Show the super fancy histogram for the simulated versus real signals for a given phantom experiment.

    Parameters
    ----------
    phantom: int or string
        Integer associated with lamp position or string with format "radius_3-digit-angle", e.g. "3_045"

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
    vessel = fig.add_axes([0.15, 0.05, 0.55, 0.55])  # [Left, Bottom, Width, Height]
    vessel.set_aspect('equal', anchor='C')
    out = fig.add_axes([0.78, 0.05, 0.2, 0.55])
    top = fig.add_axes([0.15, 0.67, 0.55, 0.2])

    phantom_profile = np.load(os.path.join(_module_path, "../phantoms-45/Phantom-%d.npy" % phantom_number))
    phantom_profile = phantom_profile.reshape((gp.n_rows, gp.n_cols))

    vessel.pcolormesh(gp.x_array_plot, gp.y_array_plot, phantom_profile)
    vessel.add_artist(plt.Circle((0., 0.), 85., color='w', fill=False))

    projections = load_projection("complex-view-cone-45.npy")[0]['projections']
    projections = projections[:32]
    f_simulated_cone, f_measured = simulate_signal(phantom_number, projections, plot=False)

    projections = load_projection("line-approximation-45.npy")[0]['projections']
    projections = projections[:32]
    f_simulated_line, f_measured = simulate_signal(phantom_number, projections, plot=False)

    top.bar(np.arange(1, 17), height=f_measured[:16], width=0.25, align='edge', label='real')
    top.bar(np.arange(1, 17) + 0.25, height=f_simulated_cone[:16], width=0.25, align='edge', label='cone of sight')
    top.bar(np.arange(1, 17) + 0.50, height=f_simulated_line[:16], width=0.25, align='edge', label='line of sight')
    top.set_ylim(0, 5)
    top.xaxis.set_major_locator(majorLocator)
    top.xaxis.set_major_formatter(majorFormatter)
    top.xaxis.set_minor_locator(minorLocator)

    out.barh(np.arange(1, 17), width=f_measured[16:], height=0.25, align='edge')
    out.barh(np.arange(1, 17) + 0.25, width=f_simulated_cone[16:], height=0.25, align='edge')
    out.barh(np.arange(1, 17) + 0.50, width=f_simulated_line[16:], height=0.25, align='edge')
    out.invert_yaxis()
    out.set_xlim(0, 5)
    out.yaxis.set_major_locator(majorLocator)
    out.yaxis.set_major_formatter(majorFormatter)
    out.yaxis.set_minor_locator(minorLocator)
    fig.legend(bbox_to_anchor=(1., 0.85))

    plt.suptitle('Phantom # %d' % phantom_number)

    plt.savefig(os.path.join(_module_path, "histograms/phantom-%d.png" % phantom_number))
    # plt.show()

    return 1


def magic_histogram(projections_dic, signals, emissivity, dead_signals=[], no_bot=False):
    """ Show the super fancy histogram for the simulated versus real signals for a given phantom experiment.

    Parameters
    ----------
    projections_dic: dictionary
        'projections': 48xNxM ndarray of the projections matrix
        'x': 1d array with x coordinates of the projections
        'y': 1d array with y coordinates of the projections
    signals: ndarray
        original signals to compare with reconstruction
    emissivity: ndarray
        emissivity function to compare with original signals
    dead_signals: iterable, optional
        list of signals no to be compared. These signals will appear in red coloring
    no_bot: bool, optional
        Set to `True` to ignore signals and projections from the bottom camera

    Returns
    -------
    ok: int
        1 if function succeeded
    """

    # orange = (251./255., 170./255., 39./255.)
    orange = (239. / 255., 123. / 255., 7. / 255.)
    blue = (0. / 255., 159. / 255., 227. / 255.)

    # Handle dead signals ----------------------------------------------------------------------------------------------
    dead_signals = np.array(dead_signals)
    dead_top = dead_signals[dead_signals < 16]
    dead_out = dead_signals[(dead_signals > 15) & (dead_signals < 32)]
    dead_bot = dead_signals[dead_signals > 31]

    x = projections_dic['x']
    y = projections_dic['y']
    projections = projections_dic['projections']

    fig = plt.figure(figsize=(8, 8))
    vessel = fig.add_axes([0.3, 0.33, 0.33, 0.33])  # [Left, Bottom, Width, Height]
    vessel.set_aspect('equal', anchor='C')
    out = fig.add_axes([0.75, 0.33, 0.2, 0.33])
    top = fig.add_axes([0.3, 0.75, 0.33, 0.2])
    if not no_bot:
        bot = fig.add_axes([0.3, 0.05, 0.33, 0.15])
    cbar_ax = fig.add_axes([0.1, 0.33, 0.05, 0.33])

    colormesh = vessel.pcolormesh(x, y, emissivity)
    vessel.add_artist(plt.Circle((0., 0.), 85., color='w', fill=False))

    vessel.set_xlabel("R (mm)", fontsize=12)
    vessel.set_ylabel("z (mm)", fontsize=12)
    vessel.set_xticks([-50, 0, 50])
    # vessel.set_xticklabels('')

    cbar = plt.colorbar(colormesh, cbar_ax)
    cbar_ax.yaxis.set_ticks_position('left')
    # cbar.set_label("a.u.", x=10)
    # cbar_ax.set_ylabel("a.u.", x=0.5)

    # Masks, negative mask: zeros inside vessel, positive mask: zeros outside vessel -------------------------------

    n_rows = len(y)
    n_cols = len(x)
    mask_radius = 85.
    res_x = np.abs(x[0] - x[1])
    res_y = np.abs(y[0] - y[1])

    ii, jj = ellipse(n_rows / 2., n_cols / 2., mask_radius / res_y, mask_radius / res_x)
    mask_negative = np.ones((n_rows, n_cols), dtype=np.float32)
    mask_negative[ii, jj] = 0.
    mask_positive = np.zeros((n_rows, n_cols), dtype=np.float32)
    mask_positive[ii, jj] = 1.

    # Apply mask to projection matrix and then reshape -------------------------------------------------------------

    P = (projections * mask_positive).reshape((projections.shape[0], -1))
    print('P:', P.shape, P.dtype)

    f_simulated = np.dot(P, emissivity.flatten())
    f_measured = signals

    bar_list = top.bar(np.arange(1, 17), height=f_measured[:16], width=0.25, align='edge', label='real', color=blue)
    for i in dead_top:
        bar_list[i].set_color('r')
    bar_list = top.bar(np.arange(1, 17) + 0.25, height=f_simulated[:16], width=0.25, align='edge', label='simulated', color=orange)
    for i in dead_top:
        bar_list[i].set_color('r')
    top.set_ylim(0, 5)
    top.set_xticks(np.arange(1, 17))
    top.set_xticklabels(['1', '', '', '', '5', '', '', '', '', '10', '', '', '', '', '15', ''])
    top.set_xlabel('detector number', fontsize=12)

    bar_list = out.barh(np.arange(1, 17), width=f_measured[16:32], height=0.25, align='edge', color=blue)
    for i in dead_out:
        bar_list[i - 16].set_color('r')
    bar_list = out.barh(np.arange(1, 17) + 0.25, width=f_simulated[16:32], height=0.25, align='edge', color=orange)
    for i in dead_out:
        bar_list[i - 16].set_color('r')
    out.invert_yaxis()
    out.set_xlim(0, 5)
    out.set_yticks(np.arange(1, 17))
    out.set_yticklabels(['1', '', '', '', '5', '', '', '', '', '10', '', '', '', '', '15', ''])
    out.set_ylabel('detector number', fontsize=12)

    if not no_bot:
        bar_list = bot.bar(np.arange(1, 17), height=f_measured[32:48], width=0.25, align='edge', color=blue)
        for i in dead_bot:
            bar_list[i - 32].set_color('r')
        bar_list = bot.bar(np.arange(1, 17) + 0.25, height=f_simulated[32:48], width=0.25, align='edge', color=orange)
        for i in dead_bot:
            bar_list[i - 32].set_color('r')
        bot.invert_yaxis()
        bot.xaxis.tick_top()
        bot.set_xticks(np.arange(1, 17))
        bot.set_xticklabels(['1', '', '', '', '5', '', '', '', '', '10', '', '', '', '', '15', ''])
        bot.set_xlabel('detector number', fontsize=12)
        bot.set_ylim(0, 5)

    fig.legend(bbox_to_anchor=(0.4, 0.5, 0.5, 0.3))

    plt.show()

    return 1
