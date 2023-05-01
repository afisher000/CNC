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
import matplotlib.pyplot as plt
import os


folder = 'simple_mammoth'
MODEL = {}

sw, ne = um.get_GPS_coords(folder)
elevations = um.get_elevation_data(sw, ne)

# Define stroke and resolution of png
DPI = 800
stroke_mm = 0.5 #mm
stroke_px = stroke_mm * (DPI/25.4)


# Generate

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


# Load contour.png
map_file = os.path.join('Projects', folder, 'contours.png')
map_img = cv.imread(map_file)
map_H, map_W = map_img.shape[:2]

full_mask = np.zeros_like(map_img[:, :, 0])
contours_to_cut = []

# Loop over hex_string colors looking for contours
for hex_string in hex_strings:
    bgr = hex_to_bgr(hex_string)
    mask = cv.inRange(map_img, bgr, bgr)
    contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    print(f'{len(contours)} contours for {hex_string}')
    full_mask = cv.bitwise_or(full_mask, mask)
    # uc.showImage(full_mask)   # Show running total of contours identified
    
    minArea = 100
    for j, contour in enumerate(contours):
        if cv.contourArea(contour)<minArea:
            continue
        # Get sorted points along contour line
        points = us.get_points_on_line(mask, contour)
        points = us.remove_duplicate_points(points, min_dist=2*stroke_px)
        points = us.sort_line_points(points, max_sep=10*stroke_px, dist_power=5, theta_buffer=0.1, max_theta = 150)
        
        # Convert pixels to CNC space, interpolate heights
        map_xpixels, map_ypixels = zip(*points)        
        xpoints = map_xpixels / (map_W-1) * (MODEL['W']) 
        ypoints = map_ypixels / (map_H-1) * (MODEL['H'])
        zpoints = um.interpolate_zpoints(xpoints, ypoints, MODEL, elevations)
            
        # Append contour
        contours_to_cut.append(list(zip(xpoints, ypoints, zpoints)))
      
        

# Order contours to reduce travel time
start_points = np.array([contour[0] for contour in contours_to_cut])
end_points = np.array([contour[-1] for contour in contours_to_cut])

# Start with first contour
idxs_to_cut = list(range(len(contours_to_cut))) # keeps track of which idxs still need cut
del idxs_to_cut[0]
contour = contours_to_cut[0]
ordered_contours = [contour] # keeps track of the ordered contours
current_point = contour[-1]

# Iterate over remaining contours
while len(idxs_to_cut)>0:
    # Find shortest path to next contour (either end)
    start_dists = norm(start_points[idxs_to_cut] - current_point, axis=1)
    end_dists = norm(end_points[idxs_to_cut] - current_point, axis=1)  
    
    if min(start_dists)<min(end_dists):
        idx = np.argmin(start_dists)
        contour = contours_to_cut[idxs_to_cut[idx]]
    else:
        idx = np.argmin(end_dists)
        contour = contours_to_cut[idxs_to_cut[idx]][::-1]
        
    # Remove from list of idxs and append. Update current point
    del idxs_to_cut[idx]
    ordered_contours.append(contour)
    current_point = contour[-1]




# %%
