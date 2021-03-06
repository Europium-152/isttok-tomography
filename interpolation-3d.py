

from scipy.interpolate import RegularGridInterpolator
import numpy as np

sensors = np.arange(30, 32)

coordinates_files = [
    "top-coordinates-sensor-1.npy",
    "top-coordinates-sensor-2.npy",
    "top-coordinates-sensor-3.npy",
    "top-coordinates-sensor-4.npy",
    "top-coordinates-sensor-5.npy",
    "top-coordinates-sensor-6.npy",
    "top-coordinates-sensor-7.npy",
    "top-coordinates-sensor-8.npy",
    "top-coordinates-sensor-9.npy",
    "top-coordinates-sensor-10.npy",
    "top-coordinates-sensor-11.npy",
    "top-coordinates-sensor-12.npy",
    "top-coordinates-sensor-13.npy",
    "top-coordinates-sensor-14.npy",
    "top-coordinates-sensor-15.npy",
    "top-coordinates-sensor-16.npy",

    "out-coordinates-sensor-1.npy",
    "out-coordinates-sensor-2.npy",
    "out-coordinates-sensor-3.npy",
    "out-coordinates-sensor-4.npy",
    "out-coordinates-sensor-5.npy",
    "out-coordinates-sensor-6.npy",
    "out-coordinates-sensor-7.npy",
    "out-coordinates-sensor-8.npy",
    "out-coordinates-sensor-9.npy",
    "out-coordinates-sensor-10.npy",
    "out-coordinates-sensor-11.npy",
    "out-coordinates-sensor-12.npy",
    "out-coordinates-sensor-13.npy",
    "out-coordinates-sensor-14.npy",
    "out-coordinates-sensor-15.npy",
    "out-coordinates-sensor-16.npy",
]

solid_angle_files = [
    # Top sensor first 8
    "solid-angle-top-and-out-sensors-1-and-16.npy",
    "solid-angle-top-and-out-sensors-2-and-15.npy",
    "solid-angle-top-and-out-sensors-3-and-14.npy",
    "solid-angle-top-and-out-sensors-4-and-13.npy",
    "solid-angle-top-and-out-sensors-5-and-12.npy",
    "solid-angle-top-and-out-sensors-6-and-11.npy",
    "solid-angle-top-and-out-sensors-7-and-10.npy",
    "solid-angle-top-and-out-sensors-8-and-9.npy",

    # Top sensor last 8
    "solid-angle-top-and-out-sensors-8-and-9.npy",
    "solid-angle-top-and-out-sensors-7-and-10.npy",
    "solid-angle-top-and-out-sensors-6-and-11.npy",
    "solid-angle-top-and-out-sensors-5-and-12.npy",
    "solid-angle-top-and-out-sensors-4-and-13.npy",
    "solid-angle-top-and-out-sensors-3-and-14.npy",
    "solid-angle-top-and-out-sensors-2-and-15.npy",
    "solid-angle-top-and-out-sensors-1-and-16.npy",

    # Outer sensor first 8
    "solid-angle-top-and-out-sensors-1-and-16.npy",
    "solid-angle-top-and-out-sensors-2-and-15.npy",
    "solid-angle-top-and-out-sensors-3-and-14.npy",
    "solid-angle-top-and-out-sensors-4-and-13.npy",
    "solid-angle-top-and-out-sensors-5-and-12.npy",
    "solid-angle-top-and-out-sensors-6-and-11.npy",
    "solid-angle-top-and-out-sensors-7-and-10.npy",
    "solid-angle-top-and-out-sensors-8-and-9.npy",

    # Outer sensor last 8
    "solid-angle-top-and-out-sensors-8-and-9.npy",
    "solid-angle-top-and-out-sensors-7-and-10.npy",
    "solid-angle-top-and-out-sensors-6-and-11.npy",
    "solid-angle-top-and-out-sensors-5-and-12.npy",
    "solid-angle-top-and-out-sensors-4-and-13.npy",
    "solid-angle-top-and-out-sensors-3-and-14.npy",
    "solid-angle-top-and-out-sensors-2-and-15.npy",
    "solid-angle-top-and-out-sensors-1-and-16.npy",

]

resolution = "0mm5"  # 0.5 millimeter resolution
# resolution = "1mm0"  # 1.0 millimeter resolution

solid_angles_dir = "D:/uni/tomography-calibration/solid-angles-values-res-" + resolution + "/"
projection_functions_dir = "D:/uni/tomography-calibration/3d-projection-functions-" + resolution + "/"
projection_matrices_dir = "D:/uni/tomography-calibration/projections/"

all_projections = []

for sensor_number in sensors:

    try:
        projection_function = np.load(projection_functions_dir + "sensor%d.npy" % sensor_number)[()]
        print("Found 3D interpolation function for sensor %d" % sensor_number)

    except IOError:

        print("3D Interpolation function for sensor %d not found" % sensor_number)
        # Solid angle values
        sangles = np.load(solid_angles_dir + solid_angle_files[sensor_number])
        # Corresponding coordinates
        coords = np.load(solid_angles_dir + coordinates_files[sensor_number])
        # Resize sangles because it has a mistake
        sangles = sangles[0:len(coords)]

        # Rounding coordinates to um. Probably not needed in 3D
        # coords = np.round(coords * 1000)
        # gridx = np.round(gridx * 1000)
        # gridy = np.round(gridy * 1000)

        # Interpolation class with projection function
        projection_function = RegularGridInterpolator((coords[:, 0], coords[:, 1], coords[:, 2]), sangles)
        np.save(projection_functions_dir + "sensor%d.npy" % sensor_number, projection_function)

    # n_rows = 128  # y-axis pixel resolution
    # n_cols = 128  # x-axis pixel resolution
    #
    # x_min = -100.
    # x_max = +100.
    #
    # y_min = -100.
    # y_max = +100.
    #
    # x_grid = np.linspace(x_min, x_max, num=n_cols)
    # y_grid = np.linspace(y_max, y_min, num=n_rows)
    #
    # projections = []
    #
    # for y in y_grid:
    #     projections.append([])
    #     for x in x_grid:
    #         projections[-1].append(projection_function(x, y)[0])
    #         # TODO: Use average value inside the grid instead of value at the center
    #
    # plt.figure()
    # plt.imshow(projections)
    # plt.colorbar()
#
#     all_projections.append(projections)
#
# all_projections = np.array(all_projections)
# np.save("projections/3d-complex-view-cone-128.npy", all_projections)

