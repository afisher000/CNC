# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 20:14:17 2023

@author: afisher
"""
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
sw, ne = um.get_GPS_coords(folder)
elevations = um.get_elevation_data(sw, ne)

# =============================================================================
# # Todo 
# Add time estimation?
# =============================================================================

# Specify model dimensions
MODEL = {}
MODEL['H'] = 150                                                               # CNC y axis
MODEL['T'] = 50                                                                 # CNC z axis
MODEL['W'] = MODEL['H'] / (ne['lat']-sw['lat'])* (ne['lng']-sw['lng'])          # CNC x axis

# Define gcode_manager
gm = gcode_manager(folder, MODEL, elevations)



# %% Contours
max_distance = .5
contours = uc.get_contours(folder, MODEL)

gm.initialize_gcode()
# For each contour, increase number of points and then interpolate z values
for contour in contours:
    new_path = uc.add_points_along_path(contour, max_distance)
    xpoints = new_path[:,0]
    ypoints = -1*new_path[:,1]
    zpoints = um.zinterpolation(xpoints, ypoints, MODEL, elevations)
    
    # Add gcode for path
    gm.append_gcode_for_path(xpoints, ypoints, zpoints)

gm.save_gcode('contours.txt')

# %% Smoothing
stepover = 0.4

gm.initialize_gcode()
xpoints, ypoints, zpoints = gm.get_raster_points(stepover)
gm.append_gcode_for_path(xpoints, ypoints, zpoints)
gm.save_gcode('smoothing.txt')



# %% Roughing stepdown
# Compute points in raster scan (higher resolution than needed to apply maximums with accuracy)
tool_diameter = 0.5 * 25.4
stepover_frac = 0.8

max_stepdown = 4
n_stepdowns = int(np.ceil(MODEL['T'] / max_stepdown))

# Get raster points
stepover = tool_diameter*stepover_frac
xpoints, ypoints, zpoints = gm.get_raster_points(stepover, roughing=True)

# Generate gcode for stepdowns
gm.initialize_gcode()
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

    
    
    
    
    
    
    
    