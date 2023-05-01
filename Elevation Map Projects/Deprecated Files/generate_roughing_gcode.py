# %%
import cv2 as cv
import numpy as np
import utils_contours as uc
import utils_smoothing as us
import utils_maps as um
from numpy.linalg import norm
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
import os





# %%
# INPUTS
folder = 'simple_mammoth'

# Define ELEVATION coordinate data
sw = {'lat':37.618210, 'lng':-119.060685}
ne = {'lat':37.656038, 'lng':-119.0001}
sw, ne = um.snap_to_GPS_points(sw, ne)

# Define fusion MODEL coordinate data
model_W = 151.9
model_H = 94.5
model_T = 30
IS_FLAT = False


# Retrieve elevation data
elevations = um.get_elevation_data(sw, ne)
min_elevation, max_elevation = np.min(elevations), np.max(elevations)
elevation_range = max_elevation - min_elevation
elev_H, elev_W = elevations.shape
elev_interp = RegularGridInterpolator((range(elev_H), range(elev_W)), elevations)

# Send to origin to start
zsafe = 5
feed_rate = 2000
gcode = f'G90 G94\nG17\nG21\nG90\nG54\nG1 Z{zsafe} F{feed_rate}\n'
gcode += f'G1 X0 Y0\n'

# Compute points in raster scan
stepover = 5
model_ys = np.arange(0, -model_H, -stepover)
model_xs = np.arange(0, model_W, stepover)
[MODEL_XS, MODEL_YS] = np.meshgrid(model_xs, model_ys)

# Compute elevation pixels
ELEV_XPIXELS = MODEL_XS * (elev_W-1) / model_W
ELEV_YPIXELS = -MODEL_YS * (elev_H-1) / model_H
ELEV_ZPIXELS = elev_interp((ELEV_YPIXELS, ELEV_XPIXELS))

# Compute model heights
MODEL_ZS = (ELEV_ZPIXELS - max_elevation)/elevation_range*model_T

# Apply maximum moving height to MODEL_ZS
reach = 2
Z_padded = np.pad(MODEL_ZS, reach, constant_values = MODEL_ZS.min())

# Find maximum of elements in each row
b = Z_padded
for row in range(Z_padded.shape[0]):
    for j in range(reach):
        b[row,:] = np.maximum(np.roll(Z_padded[row,:], 1), b[row,:])
        b[row,:] = np.maximum(np.roll(Z_padded[row,:], -1), b[row,:])

# Find maximum of elements in each column
c = Z_padded
for col in range(Z_padded.shape[1]):
    for j in range(reach):
        c[:,col] = np.maximum(np.roll(Z_padded[:,col], 1), c[:,col])
        c[:,col] = np.maximum(np.roll(Z_padded[:,col], -1), c[:,col])

# Convert back to MODEL_ZS
safety_factor = 1
MODEL_ZS = np.maximum(b, c)[reach:-reach, reach:-reach] + safety_factor

# Flip even rows of MODEL so we cut on both directions
MODEL_XS[1::2,:] = np.flip( MODEL_XS[1::2,:], axis=1)
MODEL_ZS[1::2,:] = np.flip( MODEL_ZS[1::2,:], axis=1)

# Append model points to gcode
for x, y, z in zip(MODEL_XS.ravel(), MODEL_YS.ravel(), MODEL_ZS.ravel()):
    gcode += f'G1 X{x:.2f} Y{y:.2f} Z{z:.2f}\n'

# Lift to 15mm, then head to origin
gcode += f'G1 Z15\n'
gcode += f'G1 X0 Y0\n'

#  Save gcode
gcode_path = os.path.join(folder, folder+'_gcode_roughing.txt')
with open(gcode_path, 'w') as f:
    f.write(gcode)

# %%
