# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 08:39:18 2023

@author: afisher
"""
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

# Need to add interpolation for better resolution. 

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

# Define stroke and resolution of png
DPI = 800
stroke_mm = 0.5 #mm


def gcode_for_contour(contour, zsafe=5):
    print(f'Contour Leng = {len(contour)}')
    # Generate Gcode for contour
    gcode = f'G1 X{contour[0][0]:.2f} Y{contour[0][1]:.2f}\n'
    for x,y,z in contour:
        gcode += f'G1 X{x:.2f} Y{y:.2f} Z{z:.2f}\n'
    gcode += f'G1 Z{zsafe:.2f}\n'   
    return gcode
    
def initialize_gcode(zsafe=5, feed_rate=2000):
    # Initialize gcode (used fusion example as template)
    # ORIGIN MUST BE IN TOP LEFT CORNER OF STOCK WITH Z AT 0
    # G17 -> select xy plane, G21 -> metric units, G90 -> absolute moves
    gcode = f'G90 G94\nG17\nG21\nG90\nG54\nG1 Z{zsafe} F{feed_rate}\n'
    return gcode

def dist_of_rapid_moves(contours):
    dists = []
    for j in range(len(contours)-1):
        dists.append(norm(np.array(contours[j+1][0])-np.array(contours[j][-1])))
    return sum(dists), dists

def hex_to_bgr(hex_string):
    ''' Convert a hexadecimal color string to an bgr array'''
    hex_value = hex_string.lstrip('#')
    bgr = np.array([int(hex_value[i:i+2], 16) for i in (4, 2, 0)])
    return bgr

# Basic colors
hex_strings = [
    '#ff0000', #red
    '#00ff00', #green
    '#0000ff', #blue
    '#ffff00', #yellow
    '#ff6600', #orange
    '#000000', #black
]


# Retrieve elevation data
elevations = um.get_elevation_data(sw, ne)
min_elevation, max_elevation = np.min(elevations), np.max(elevations)
elevation_range = max_elevation - min_elevation
elev_H, elev_W = elevations.shape
elev_interp = RegularGridInterpolator((range(elev_H), range(elev_W)), elevations)


# Compute stroke in pixels
stroke_px = stroke_mm * (DPI/25.4)

# Read contours
map_file = os.path.join(folder, 'contours.png')
map_img = cv.imread(map_file)
map_H, map_W = map_img.shape[:2]

full_mask = np.zeros_like(map_img[:, :, 0])
contours_to_cut = []

# Analyze image
for hex_string in hex_strings:
    bgr = hex_to_bgr(hex_string)
    mask = cv.inRange(map_img, bgr, bgr)
    contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    print(f'{len(contours)} contours for {hex_string}')
    full_mask = cv.bitwise_or(full_mask, mask)
    # uc.showImage(full_mask)
    
    minArea = 100
    # For each contour, find MAP points along contour
    for j, contour in enumerate(contours):
        if cv.contourArea(contour)<minArea:
            continue
        points = us.get_points_on_line(mask, contour)
        points = us.remove_duplicate_points(points, min_dist=2*stroke_px)
        points = us.sort_line_points(points, max_sep=10*stroke_px, dist_power=5, theta_buffer=0.1, max_theta = 150)
        
        # Convert MAP points to ELEVATION points
        map_xpixels, map_ypixels = zip(*points)
        elev_xpixels = np.array(map_xpixels) / (map_W-1) * (elev_W-1)
        elev_ypixels = np.array(map_ypixels) / (map_H-1) * (elev_H-1)
        elev_zpixels = elev_interp((elev_ypixels, elev_xpixels)) # Don't like this termed pixels... Its not, its exact
        
        # Convert ELEVATION points to MODEL points
        model_xs = elev_xpixels / (elev_W-1) * model_W
        model_ys = -elev_ypixels / (elev_H-1) * model_H
        if IS_FLAT:
            cut_depth = 1
            model_zs = -cut_depth * np.ones_like(elev_zpixels)
        else:
            model_zs = (elev_zpixels-max_elevation)/elevation_range*model_T
            
        # Append contour
        contours_to_cut.append(list(zip(model_xs, model_ys, model_zs)))
      
        


start_points = np.array([contour[0] for contour in contours_to_cut])
end_points = np.array([contour[-1] for contour in contours_to_cut])


# Start with first contour
idxs_to_cut = list(range(len(contours_to_cut)))
del idxs_to_cut[0]
contour = contours_to_cut[0]
ordered_contours = [contour]
current_point = contour[-1]

# Iterate over remaining contours
while len(idxs_to_cut)>0:
    start_dists = norm(start_points[idxs_to_cut] - current_point, axis=1)
    end_dists = norm(end_points[idxs_to_cut] - current_point, axis=1)    
    if min(start_dists)<min(end_dists):
        idx = np.argmin(start_dists)
        contour = contours_to_cut[idxs_to_cut[idx]]
    else:
        idx = np.argmin(end_dists)
        contour = contours_to_cut[idxs_to_cut[idx]][::-1]
    del idxs_to_cut[idx]
    ordered_contours.append(contour)
    current_point = contour[-1]




# Generate gcode
gcode = initialize_gcode()
for contour in ordered_contours:
    gcode += gcode_for_contour(contour)


print(f'Unordered = {dist_of_rapid_moves(contours_to_cut)[0]}')
print(f'Ordered = {dist_of_rapid_moves(ordered_contours)[0]}')
#  # Save to file
gcode_path = os.path.join(folder, folder+'_gcode_contours.txt')
with open(gcode_path, 'w') as f:
    f.write(gcode)

    
def get_raster_points(stepover):
    model_ys = np.arange(0, -model_H, -stepover)
    model_xs = np.arange(0, model_W, stepover)
    [MODEL_XS, MODEL_YS] = np.meshgrid(model_xs, model_ys)

    # Flip even rows of MODEL_XS so we cut on both directions
    MODEL_XS[1::2,:] = np.flip( MODEL_XS[1::2,:], axis=1)


    # Compute elevation pixels
    ELEV_XPIXELS = MODEL_XS * (elev_W-1) / model_W
    ELEV_YPIXELS = -MODEL_YS * (elev_H-1) / model_H
    ELEV_ZPIXELS = elev_interp((ELEV_YPIXELS, ELEV_XPIXELS))

    # Compute model heights
    MODEL_ZS = (ELEV_ZPIXELS - max_elevation)/elevation_range*model_T
    return MODEL_XS, MODEL_YS, MODEL_ZS



def generate_smoothing_gcode():

    # Send to origin to start
    gcode = initialize_gcode()
    gcode += f'G1 X0 Y0\n'

    # Compute points in raster scan
    stepover = 0.4
    MODEL_XS, MODEL_YS, MODEL_ZS = get_raster_points(stepover)


    # Append model points to gcode
    for x, y, z in zip(MODEL_XS.ravel(), MODEL_YS.ravel(), MODEL_ZS.ravel()):
        gcode += f'G1 X{x:.2f} Y{y:.2f} Z{z:.2f}\n'


    # Lift to 15mm, then head to origin
    gcode += f'G1 Z15\n'
    gcode += f'G1 X0 Y0\n'


    #  Save gcode
    gcode_path = os.path.join(folder, folder+'_gcode_smoothing.txt')
    with open(gcode_path, 'w') as f:
        f.write(gcode)
    
generate_smoothing_gcode()


def compute_2d_rolling_maximum(arr, reach, safety_factor=1):
    arr_padded = np.pad(arr, reach, constant_values = arr.min())

    # Find maximum of elements in each row
    b = arr_padded
    for row in range(arr_padded.shape[0]):
        for _ in range(reach):
            b[row,:] = np.maximum(np.roll(arr_padded[row,:], 1), b[row,:])
            b[row,:] = np.maximum(np.roll(arr_padded[row,:], -1), b[row,:])

    # Find maximum of elements in each column
    c = arr_padded
    for col in range(arr_padded.shape[1]):
        for _ in range(reach):
            c[:,col] = np.maximum(np.roll(arr_padded[:,col], 1), c[:,col])
            c[:,col] = np.maximum(np.roll(arr_padded[:,col], -1), c[:,col])

    # Convert back to MODEL_ZS
    safety_factor = 1
    return np.maximum(b, c)[reach:-reach, reach:-reach] + safety_factor





def generate_roughing_gcode():
    # Send to origin to start
    gcode = initialize_gcode()
    gcode += f'G1 X0 Y0\n'

    # Compute points in raster scan (higher resolution than needed to apply maximums with accuracy)
    stepover = 5
    res_factor = 4
    grid_res = stepover/res_factor
    MODEL_XS, MODEL_YS, MODEL_ZS = get_raster_points(grid_res)

    # Apply maximum moving height to MODEL_ZS
    tool_radius = 0.25*25.4  * (1/2)
    reach = int(np.ceil(tool_radius/grid_res))
    MODEL_ZS = compute_2d_rolling_maximum(MODEL_ZS, reach)

    # Downsample to correct stepover
    MODEL_XS = MODEL_XS[::res_factor, ::res_factor]
    MODEL_YS = MODEL_YS[::res_factor, ::res_factor]
    MODEL_ZS = MODEL_ZS[::res_factor, ::res_factor]
    
    # Flip even rows of MODEL so we cut on both directions
    MODEL_XS[1::2,:] = np.flip( MODEL_XS[1::2,:], axis=1)
    MODEL_ZS[1::2,:] = np.flip( MODEL_ZS[1::2,:], axis=1)


    # Lift to 15mm, then head to origin
    gcode += f'G1 Z15\n'
    gcode += f'G1 X0 Y0\n'

    roughing_stepdown = 4
    xstart = -10
    n_stepdowns = int(np.ceil(model_T / roughing_stepdown))
    for j in range(n_stepdowns):
        # Compute ZS for current stepdown
        stepdown_ZS = MODEL_ZS * (j+1)/n_stepdowns 

        # Move to off stock, then move to starting height
        gcode += f'G1 X{xstart} Y0\n'
        gcode += f'G1 Z{stepdown_ZS[0,0]}\n'
        
        # Append model points to gcode
        for x, y, z in zip(MODEL_XS.ravel(), MODEL_YS.ravel(), stepdown_ZS.ravel()):
            gcode += f'G1 X{x:.2f} Y{y:.2f} Z{z:.2f}\n'

        # Lift to 15mm, then head to origin
        gcode += f'G1 Z15\n'
        gcode += f'G1 X0 Y0\n'

    #  Save gcode
    gcode_path = os.path.join(folder, folder+'_gcode_roughing.txt')
    with open(gcode_path, 'w') as f:
        f.write(gcode)



generate_roughing_gcode()
# %%

# %%
