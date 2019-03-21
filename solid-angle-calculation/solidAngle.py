"""
Solid Angle
"""
from scipy.integrate import dblquad
import numpy as np
import time


# PIXEL COORDINATES (x,z)


class Rotation:
    """Performs a rotation around zz axis with given angle and a translation
    
    Parameters:
        angle: float
            Rotation angle in degrees
        trans: iterable (3 floats)
            Translation vector. Translation is performed after rotation  
            
    Methods:
        3x1 numpy.array rotate(x):
            x: 3x1 numpy.array
                Vector to be transformed with the rotation and translation specified at instantiation
    """

    def __init__(self, angle, trans):
        self.theta = np.radians(angle)
        self.matrix = np.array(
            [[np.cos(self.theta), -np.sin(self.theta), 0.], [np.sin(self.theta), np.cos(self.theta), 0.], [0., 0., 1.]])
        self.translation = np.array(trans)

    def rotate(self, x):
        return self.matrix.dot(x) + self.translation


rotate_top = Rotation(180., [5., 97., 0])
rotate_out = Rotation(90., [109., 0., 0.])
rotate_bot = Rotation(7.5, [5. + 102 * np.cos(np.radians(-82.5)), 102.0 * np.sin(np.radians(-82.5)), 0.])

# xx bounding coordinates of the sensors on the pinhole frame. Starting from the 1st
xMinMax = [
    [6.75, 7.5],
    [5.8, 6.55],
    [4.85, 5.6],
    [3.9, 4.65],
    [2.95, 3.7],
    [2.0, 2.75],
    [1.05, 1.8],
    [0.1, 0.85],
    #        [-0.85,-0.1],
    #        [-1.8,-1.05],
    #        [-2.75,-2.0],
    #        [-3.7,-2.95],
    #        [-4.65,-3.9],
    #        [-5.6,-4.85],
    #        [-6.55,-5.8],
    #        [-7.5,-6.75]
]

sensors_index = [1, 2, 3, 4, 5, 6, 7, 8]

# zz bounding coordinates for the sensors
zMin = -2.025
zMax = 2.025

# yy coordinate of the sensors. This is the distance between pinhole and sensors
h = -9.

# Pinhole Radius
radius = 0.4

# Grid parameters for computation
Xstart = -100
Xend = 100
Xstep = 1

Ystart = 1.
Yend = 204.
Ystep = 1

Zstart = 0.
Zend = 100.
Zstep = 1

Xpoints = int((Xend - Xstart) / Xstep) + 1
Ypoints = int((Yend - Ystart) / Ystep) + 1
Zpoints = int((Zend - Zstart) / Zstep) + 1

# ------------------------------------------------------------------------
# Early declaration of the arrays to hold the data
# ------------------------------------------------------------------------

# Solid angle of each point in space defined by the grid
solid_angles = np.zeros(Xpoints * Ypoints * Zpoints)
# Integration error
solid_angles_errors = np.zeros(Xpoints * Ypoints * Zpoints)

# Coordinates of each point where the solid angle is computed on the pinhole frame
# All points which are not computed have either zero solid angle of are out of the computation grid
coordinates = np.zeros((Xpoints * Ypoints * Zpoints, 3))
# Transformed coordinates for the top camera
top = np.zeros((Xpoints * Ypoints * Zpoints, 3))
# Transformed coordinates for the bottom camera
out = np.zeros((Xpoints * Ypoints * Zpoints, 3))

# Point (x,y,z) on sensor (i) is the same as point (-x,y,z) on sensor (n-i)
top_mirror = np.zeros((Xpoints * Ypoints * Zpoints, 3))
out_mirror = np.zeros((Xpoints * Ypoints * Zpoints, 3))

# Coordinates of every point on the 2D grid (x and y). Tokamak cross section
coords = np.mgrid[Xstart:(Xend + Xstep):Xstep, Ystart:(Yend + Ystep):Ystep]
x_coord = coords[0].flatten()
y_coord = coords[1].flatten()

# xy = np.mgrid[Xstart:(Xend+Xstep):Xstep,Ystart:(Yend+Ystep):Ystep].reshape(2,-1).T


print("Solid angles over a matrix: (%d, %d, %d)" % (Xpoints, Ypoints, Zpoints))

print("Total points: %d" % (Xpoints * Ypoints * Zpoints))

px = xMinMax[0]

# %%
t0 = time.time()
for px, sensor in zip(xMinMax, sensors_index):
    # Start timer of the calculation

    # Low xx side of the rectangles
    rectXmin = x_coord - y_coord * (px[0] - x_coord) / (h - y_coord)

    # High xx side of the rectangles
    rectXmax = x_coord - y_coord * (px[1] - x_coord) / (h - y_coord)

    # Array of radii of the pinhole needed for comparison reasons
    comparable = radius * np.ones(len(rectXmax))

    # The surface xx coordinate starts on the MAXIMUM between the begining of the pinhole 
    # and the begining of the rectangle
    lowXlim = np.maximum(rectXmin, -comparable)

    # The surface xx coordinate ENDS on the MINIMUM between the end of the pinhole 
    # and the end of the rectangle
    upXlim = np.minimum(rectXmax, comparable)

    # Conditional Index. We only compute points for which the lowXlim is smaller than the upXlim.
    # Only such points have a visual on the sensor for z=0 
    cond_idx = upXlim > lowXlim

    # Index to count where to store the values of the calculated solid angles
    index = 0
    for x0, y0, lowXlimit, upXlimit in zip(x_coord[cond_idx], y_coord[cond_idx], lowXlim[cond_idx], upXlim[cond_idx]):

        for z0 in np.arange(Zstart, Zend + Zstep, Zstep):

            # zz boundaries of the rectangle
            startZ = z0 - y0 * (zMin - z0) / (h - y0)
            endZ = z0 - y0 * (zMax - z0) / (h - y0)

            # Integration is done on the xz plane
            # For each value of x, calculate the z boundaries

            def lowZlim(x):

                # Pinhole boundaries for given x
                low_circle = -np.sqrt(radius ** 2 - x ** 2)
                up_circle = np.sqrt(radius ** 2 - x ** 2)

                # If the rectangle ends before the pinhole begins of vice-versa
                # Return zero for the integral to also yield zero
                if (low_circle > endZ or up_circle < startZ):
                    return 0.0

                # Otherwise, the lower limit for integration is the maximum between
                # the begining of the pinhole and the rectangle
                else:
                    return max((startZ, low_circle))


            def upZlim(x):

                low_circle = -np.sqrt(radius ** 2 - x ** 2)
                up_circle = np.sqrt(radius ** 2 - x ** 2)

                if (low_circle > endZ or up_circle < startZ):
                    return 0.0

                else:
                    return min((endZ, up_circle))


            s_angle, s_error = \
                dblquad(lambda z, x: y0 / ((x - x0) * (x - x0) + y0 * y0 + (z - z0) * (z - z0)) ** (3. / 2.),
                        lowXlimit, upXlimit,
                        lowZlim, upZlim,
                        args=(), epsrel=1.49e-8)

            if s_angle > 0:
                top[index] = rotate_top.rotate(np.array([x0, y0, z0]))
                out[index] = rotate_out.rotate(np.array([x0, y0, z0]))

                top_mirror[index] = rotate_top.rotate(np.array([-x0, y0, z0]))
                out_mirror[index] = rotate_out.rotate(np.array([-x0, y0, z0]))

                solid_angles[index] = s_angle
                solid_angles_errors[index] = s_error

                index += 1

            #                print("%.12f pm %.12f (x10e-5)" % (s_angle*1e5,s_error*1e5) )

            # If the solid angle is zero, we've exited the view cone completly.
            else:
                break

    np.save("top-coordinates-sensor-%d.npy" % sensor, top[0:index])
    np.save(("top-coordinates-sensor-%d.npy" % (17 - sensor)), top_mirror[0:index])

    np.save("out-coordinates-sensor-%d.npy" % sensor, out[0:index])
    np.save("out-coordinates-sensor-%d.npy" % (17 - sensor), out_mirror[0:index])

    np.save("solid-angle-top-and-out-sensors-%d-and-%d.npy" % (sensor, 17 - sensor), solid_angles)
    np.save("error-sa-top-and-out-sensors-%d-and-%d.npy" % (sensor, 17 - sensor), solid_angles_errors)

t1 = time.time()

total = t1 - t0

print("computation time: %f\ncalculations: %d\ntime per calculation: %f" % (
total, Xpoints * Ypoints * Zpoints, total / (Xpoints * Ypoints * Zpoints)))
