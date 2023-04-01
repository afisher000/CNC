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


# Define ELEVATION coordinate data
sw = {'lat':47.267153, 'lng':-120.4398904}
ne = {'lat':47.297119, 'lng':-120.391744}
elevations = um.get_elevation_data(sw, ne)
min_elevation, max_elevation = np.min(elevations), np.max(elevations)
elevation_range = max_elevation- min_elevation
elev_H, elev_W = elevations.shape
elev_interp = RegularGridInterpolator((range(elev_H), range(elev_W)), elevations)


# Define fusion MODEL coordinate data
model_W = 137.6 
model_H = 84.8
model_T = 20
IS_FLAT = False
    
# Initialize gcode (use fusion example as template)
# ORIGIN MUST BE IN TOP LEFT CORNER OF STOCK WITH Z AT 0
# G17 -> select xy plane, G21 -> metric units, G90 -> absolute moves
feed_rate = 2000
zsafe = 5
cut_depth = 1
gcode = f'G90 G94\nG17\nG21\nG90\nG54\nG1 Z{zsafe} F{feed_rate}\n'



# Basic colors
hex_strings = [
    '#ff0000', #red
    '#00ff00', #green
    '#0000ff', #blue
    '#ffff00', #yellow
    '#ff6600', #orange
    '#000000', #black
]



def hex_to_bgr(hex_string):
    ''' Convert a hexadecimal color string to an bgr array'''
    hex_value = hex_string.lstrip('#')
    bgr = np.array([int(hex_value[i:i+2], 16) for i in (4, 2, 0)])
    return bgr


DPI = 800
stroke_mm = 0.5 #mm
stroke_px = stroke_mm * (DPI/25.4)


folder = 'missionridge'
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
        elev_xpixels = np.array(map_xpixels) / map_W * elev_W
        elev_ypixels = np.array(map_ypixels) / map_H * elev_H
        elev_zpixels = elev_interp((elev_ypixels, elev_xpixels))
        
        # Convert ELEVATION points to MODEL points
        model_xs = elev_xpixels / elev_W * model_W
        model_ys = -elev_ypixels / elev_H * model_H
        if IS_FLAT:
            model_zs = -cut_depth * np.ones_like(elev_zpixels)
        else:
            model_zs = (elev_zpixels-max_elevation)/elevation_range*model_T
            
        # Append contour
        contours_to_cut.append(list(zip(model_xs, model_ys, model_zs)))
      
        
def gcode_for_contour(contour):
    # Generate Gcode for contour
    gcode = f'G1 X{contour[0][0]} Y{contour[0][1]}\n'
    for x,y,z in contour:
        gcode += f'G1 X{x} Y{y} Z{z}\n'
    gcode += f'G1 Z{zsafe}\n'   
    return gcode

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


def dist_of_rapid_moves(contours):
    dists = []
    for j in range(len(contours)-1):
        dists.append(norm(np.array(contours[j+1][0])-np.array(contours[j][-1])))
    return sum(dists), dists

# Generate gcode
for contour in ordered_contours:
    gcode += gcode_for_contour(contour)


print(f'Unordered = {dist_of_rapid_moves(contours_to_cut)[0]}')
print(f'Ordered = {dist_of_rapid_moves(ordered_contours)[0]}')
#  # Save to file
gcode_path = os.path.join(folder, folder+'_gcode.txt')
with open(gcode_path, 'w') as f:
    f.write(gcode)
    

