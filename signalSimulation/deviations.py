import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append("D:/uni/tomography-calibration")
import globalParameters as gp
from calibrationShots import keys
from simulateSignals import simulate_signal


absolute_deviation = [[] for i in range(32)]
relative_deviation = [[] for i in range(32)]

# phantoms-45-gaussian = [2]
phantoms = np.arange(33)
# phantoms-45-gaussian = np.arange(9)

plot = False
# plot = True

for phantom in phantoms:
    #
    # for reconstruction_time in np.linspace(15900, 23000, 3):
        # phantom = 3

    try:
        phantom_number = keys.index(phantom)
    except ValueError:
        phantom_number = phantom

    if plot:
        fig = plt.figure(figsize=(8, 8))
        vessel = fig.add_axes([0.05, 0.05, 0.4, 0.4])  # [Left, Bottom, Width, Height]
        vessel.set_aspect('equal', anchor='C')
        out = fig.add_axes([0.55, 0.05, 0.4, 0.4])
        top = fig.add_axes([0.05, 0.55, 0.4, 0.4])

    phantom_profile = np.load("../phantoms-45-gaussian/Phantom-%d.npy" % phantom_number)
    phantom_profile = phantom_profile.reshape((gp.n_rows, gp.n_cols))

    if plot:
        vessel.pcolormesh(gp.x_array_plot, gp.y_array_plot, phantom_profile)
        vessel.add_artist(plt.Circle((0., 0.), 85., color='w', fill=False))

    f_simulated, f_measured = simulate_signal(phantom_number, reconstruction_time='auto', plot=False)

    # Trim down to ignore reflections
    sensors = np.arange(32)
    sensors = sensors[f_simulated > 0.001]

    for i in sensors:
        absolute_deviation[i].append(f_measured[i] - f_simulated[i])
        relative_deviation[i].append((f_measured[i] - f_simulated[i]) / f_simulated[i])

    if plot:
        top.bar(np.arange(1, 17), height=f_simulated[:16], width=0.3, align='edge', label='simulated')
        top.bar(np.arange(1, 17) + 0.3, height=f_measured[:16], width=0.3, align='edge', label='real')
        top.set_ylim(0, 4)

        out.barh(np.arange(1, 17), width=f_simulated[16:], height=0.3, align='edge', label='simulated')
        out.barh(np.arange(1, 17) + 0.3, width=f_measured[16:], height=0.3, align='edge', label='real')
        out.invert_yaxis()
        out.set_xlim(0, 4)

        plt.savefig("D:/desktop/phantoms-45-gaussian/%d.png" % phantom)

sensor = 26
plt.figure()
plt.hist(absolute_deviation[sensor])
plt.title("Absolute Deviation for sensor %d\nStatistics #%d" % (sensor, len(absolute_deviation[sensor])))
plt.xlabel("Volt")
plt.show()

plt.figure()
plt.hist(np.array(relative_deviation[sensor])*100)
plt.title("Relative Deviation for sensor %d\nStatistics #%d" % (sensor, len(absolute_deviation[sensor])))
plt.xlabel("%")
plt.show()

