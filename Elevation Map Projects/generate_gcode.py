# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 20:14:17 2023

@author: afisher
"""
# %%
import cv2 as cv
import numpy as np
import utils_contours as uc
import utils_maps as um
from gcode_manager import gcode_manager
from numpy.linalg import norm
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
import os


folder = 'simple_mammoth'
# sw, ne = um.get_GPS_coords(folder)
# elevations = um.get_elevation_data(sw, ne)

# =============================================================================
# # Todo 
# Add time estimation?
# =============================================================================

# Specify model dimensions
model = {}
model['H'] = 150                                                               # CNC y axis
model['T'] = 50                                                                 # CNC z axis

# Define gcode_manager
gm = gcode_manager(folder, model)



# %% Contours
# Define parameters
max_distance = .5                   # Max distance between points on contour
transition_height = 5               # Height above stock when making transition

# Get paths from contours.svg
paths = uc.get_svg_paths(folder, model)

# Loop over paths 
gm.initialize_gcode()
for j in range(len(paths)):
    path = paths[j]

    # Add points to satisfy max_distance, unpack x and y. Switch sign on y for different coordinate system.
    new_path = uc.add_points_along_path(path, max_distance)
    xpoints = new_path[:,0]
    ypoints = -1*new_path[:,1]
    
    # Append gcode for path
    gm.append_gcode_for_path(xpoints, ypoints, lift=False, zdepth=-1, is_contour=True)

    # Add transition between end of current path and beginning of next path
    if j<(len(paths)-1):
        transition = np.vstack([paths[j][-1], paths[j+1][0]])
        new_path = uc.add_points_along_path(transition, max_distance)
        xpoints = new_path[:,0]
        ypoints = -1*new_path[:,1]

        # Add gcode for transition, bit will be at transition_height.
        gm.append_gcode_for_path(xpoints, ypoints, lift=False, zdepth = transition_height)

gm.save_gcode('paths.txt')

# %% Smoothing
# Inputs
stepover = 0.4



gm.initialize_gcode()

# Get raster points
xpoints, ypoints, zpoints = gm.get_raster_points(stepover)

# Specify zpoints 
gm.append_gcode_for_path(xpoints, ypoints, zpoints=zpoints)
gm.save_gcode('smoothing.txt')



# %% Roughing stepdown
tool_diameter = 0.5 * 25.4
stepover_frac = 0.8
max_stepdown = 4 #in mm

# Get raster points
stepover = tool_diameter*stepover_frac
xpoints, ypoints, zpoints = gm.get_raster_points(stepover, roughing=True)

# Generate gcode for each stepdown height
gm.initialize_gcode()
n_stepdowns = int(np.ceil(model['T'] / max_stepdown))
for j in range(n_stepdowns):
    gm.append_gcode_for_path(xpoints, ypoints, zpoints*(j+1)/n_stepdowns , roughing=True)

gm.save_gcode('roughing_stepdowns.txt')


# %% Roughing single pass
tool_diameter = 1/4 * 25.4
stepover_frac = 0.8


gm.initialize_gcode()
stepover = tool_diameter*stepover_frac
xpoints, ypoints, zpoints = gm.get_raster_points(stepover)
gm.append_gcode_for_path(xpoints, ypoints, zpoints)
gm.save_gcode('roughing.txt')

    
    
    
    
    
    
    
    