# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 20:14:17 2023

@author: afisher
"""
# %%
from gcode_manager import gcode_manager


# =============================================================================
# # Todo 
# Logo input: Input dimensions of the logo and padding (specify left/right padding to 
# place inside.)
# Need to keep track of max/min xy positions of the logo.
# =============================================================================

# Specify folder and model dimensions
folder = 'mammoth'
model = {}
model['H'] = 150                                                               # CNC y axis
model['T'] = 50                                                                 # CNC z axis

# Define gcode_manager
gm = gcode_manager(folder, model)

# gm.generate_roughing_gcode('roughing.txt', tool_diameter = 0.25*25.4, max_stepdown=4)
# gm.generate_smoothing_gcode('smoothing.txt', tool_diameter=4, stepover=1)
# gm.generate_runs_gcode('runs.txt', zdepth=-1)
gm.generate_image_gcode('signature')
