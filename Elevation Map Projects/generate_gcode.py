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

# Specify model dimensions'
model = {}
model['H'] = 150                                                               # CNC y axis
model['T'] = 50                                                                 # CNC z axis

# Define gcode_manager
gm = gcode_manager(folder, model)

gm.generate_roughing_gcode('roughing.txt', tool_diameter = 0.25*25.4, max_stepdown=4)
gm.generate_smoothing_gcode('smoothing.txt', tool_diameter=4, stepover=1)
gm.generate_runs_gcode('runs.txt', zdepth=-1)
