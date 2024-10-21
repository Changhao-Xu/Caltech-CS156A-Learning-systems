import math
import numpy as np

gd_thresh = 10e-14
gd_lrate = 0.1
gd_start = np.array([1,1])
gd_coord_iters = 15

def gd_error(cur_coords): # calculate current error function value
    u = cur_coords[0]
    v = cur_coords[1]
    cur_error = math.pow(u*math.pow(math.e, v) - 2.0*v*math.pow(math.e, -1 * u), 2.0)
    return cur_error

def gd_partial(cur_coords): # calculate partial derivative
    u = cur_coords[0]
    v = cur_coords[1]
    u_coord = 2.0*(math.pow(math.e, v) + 2.0*v*math.pow(math.e, -1 * u)) * (u*math.pow(math.e,v) - 2.0*v*math.pow(math.e, -1 * u))
    v_coord = 2.0*(u*math.pow(math.e, v) - 2.0*math.pow(math.e, -1 * u))*(u*math.pow(math.e, v) - 2*v*math.pow(math.e, -1 * u))
    return np.array([u_coord, v_coord])

def gd_perform(coords, l_rate, threshold):
    num_iterations = 0
    cur_error = gd_error(coords)
    while cur_error >= threshold:
        num_iterations = num_iterations + 1
        cur_partial = gd_partial(coords)
        coords = np.subtract(coords, np.multiply(l_rate, cur_partial))
        cur_error = gd_error(coords)
    return coords, cur_error, num_iterations

def gd_coord_perform(coords, l_rate, iters):
    #instead of moving along both coords, moving along u, then v, then u, then v...
    cur_error = 0
    for cur_iter in range(iters):
        cur_partial = gd_partial(coords)
        coords = np.subtract(coords, np.multiply(l_rate, np.multiply(np.array([1,0]), cur_partial))) # first move along u coord so 0 out v coord
        cur_partial = gd_partial(coords)
        coords = np.subtract(coords, np.multiply(l_rate, np.multiply(np.array([0,1]), cur_partial))) # now redo for v coord
        cur_error = gd_error(coords)
    return cur_error


gd_coords, gd_err, gd_numiters = gd_perform(gd_start, gd_lrate, gd_thresh)
gd_cerr = gd_coord_perform(gd_start, gd_lrate, gd_coord_iters)
print("With starting coordinates (%f,%f), learning rate %f and threshold %e:" % (gd_start[0], gd_start[1], gd_lrate, gd_thresh))
print("It took %d iterations to achieve an error of %e ending at coordinates (%f,%f)." % (gd_numiters, gd_err, gd_coords[0], gd_coords[1]))
print("Iterating coordinate-wise for %d iterations, the resulting error is %f." % (gd_coord_iters, gd_cerr))
