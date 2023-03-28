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



# Define ELEVATION coordinate data
sw = {'lat':47.267153, 'lng':-120.4398904}
ne = {'lat':47.297119, 'lng':-120.391744}
elevations = um.get_elevation_data(sw, ne)
min_elevation, max_elevation = np.min(elevations), np.max(elevations)
elevation_range = max_elevation- min_elevation
elev_H, elev_W = elevations.shape
elev_interp = RegularGridInterpolator((range(elev_H), range(elev_W)), elevations)

# Define fusion MODEL coordinate data
model_W = 50
model_H = 100
model_T = 30
    
# Initialize gcode (use fusion example as template)
# ORIGIN MUST BE IN TOP LEFT CORNER OF STOCK WITH Z AT 0
# G17 -> select xy plane, G21 -> metric units, G90 -> absolute moves
feed_rate = 2000
zsafe = 5
gcode = f'G90 G94\nG17\nG21\nG28 G91 Z0\nG90\nG54\nG0 Z{zsafe} F{feed_rate}\n'


# Basic colors in BGR format
rgbs = {
    'green':np.array([76, 177, 34]),
    'blue':np.array([204, 72, 63]),
    'black':np.array([0,0,0]),
    'red':np.array([36, 28, 237]),
    'orange':np.array([39, 127, 255]),
    'gray':np.array([127,127,127]),
    'yellow':np.array([0, 242, 255])
}

map_file = 'colors.png'
map_img = cv.imread(map_file)
map_H, map_W = map_img.shape[:2]



# Analyze image
for color, rgb in rgbs.items():
    mask = cv.inRange(map_img, rgb, rgb)
    contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    print(f'{len(contours)} contours for {color}')
    # uc.showImage(mask)
    # For each contour, find MAP points along contour
    for j, contour in enumerate(contours):
        points = us.get_points_on_line(mask, contour)
        points = us.remove_duplicate_points(points, min_dist=5)
        points = us.sort_line_points(points, max_sep=80, dist_power=5, theta_buffer=0.1, max_theta = 150)
      
        # Convert MAP points to ELEVATION points
        map_xpixels, map_ypixels = zip(*points)
        elev_xpixels = np.array(map_xpixels) / map_W * elev_W
        elev_ypixels = np.array(map_ypixels) / map_H * elev_H
        elev_zpixels = elev_interp((elev_ypixels, elev_xpixels))
        
        # Convert ELEVATION points to MODEL points
        model_xs = elev_xpixels / elev_W * model_W
        model_ys = -elev_ypixels / elev_H * model_H
        model_zs = (elev_zpixels-max_elevation)/elevation_range*model_T
        
        # Generate Gcode for contour
        gcode += f'G0 X{model_xs[0]} Y{model_ys[0]}\n'
        for x,y,z in zip(model_xs, model_ys, model_zs):
            gcode += f'G0 X{x} Y{y} Z{z}\n'
        gcode += f'G0 Z{zsafe}'
    
 # Save to file
with open('test.txt', 'w') as f:
    f.write(gcode)
    

