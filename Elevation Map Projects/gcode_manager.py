# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 18:14:53 2023

@author: afisher
"""
import os
import numpy as np
from utils_maps import zinterpolation

class gcode_manager():
    def __init__(self, folder, MODEL, elevations):
        self.folder = folder
        self.MODEL = MODEL
        self.elevations = elevations
        self.gcode = ''
        self.zsafe = 5
        self.feed_rate = 2000
        self.xsafe = -20
        self.roughing_factor = 4        
        return
    
    
    def save_gcode(self, file):
        path = os.path.join('Projects', self.folder, file)
        with open(path, 'w') as f:
            f.write(self.gcode)
        print(f'Saved "{file}"')
        
        
    def append_gcode_for_path(self, xpoints, ypoints, zpoints, roughing=False):
        # Go to xy starting point
        self.gcode += f'G1 X{xpoints[0]:.2f} Y{ypoints[0]:.2f}\n'
        
        # If roughing, start off stock at correct z depth
        if roughing:
            self.gcode += f'G1 X{self.xsafe:.2f} Y0\n'
            self.gcode += f'G1 Z{zpoints[0]:.2f}\n'
        
        # Move along points in path
        lines = [f'G1 X{x:.2f} Y{y:.2f} Z{z:.2f}\n' for x,y,z in
                 zip(xpoints, ypoints, zpoints)]
        self.gcode += ''.join(lines)
            
        # List to safe height
        self.gcode += f'G1 Z{self.zsafe:.2f}\n'   
        return
    
    
    
    def initialize_gcode(self):
        # Initialize gcode (used fusion example as template)
        # ORIGIN MUST BE IN TOP LEFT CORNER OF STOCK WITH Z AT 0
        # G17 -> select xy plane, G21 -> metric units, G90 -> absolute moves
        self.gcode = f'G90 G94\nG17\nG21\nG90\nG54\nG1 Z{self.zsafe} F{self.feed_rate}\n'
    
    
    
    def get_raster_points(self, stepover, roughing=False):
        
        # If roughing, decrease stepover to apply rolling maximum
        if roughing:
            stepover = stepover/self.roughing_factor
        
        unique_y = np.arange(0, -self.MODEL['H'], -stepover)
        unique_x = np.arange(0, self.MODEL['W'], stepover)
        [xpoints, ypoints] = np.meshgrid(unique_x, unique_y)
    
        # Interpolate zpoints
        zpoints = zinterpolation(xpoints, ypoints, self.MODEL, self.elevations)
        
        # Apply rolling maximum, downsample back to desired stepover
        if roughing:
            zpoints = self.compute_2d_rolling_maximum(zpoints, self.roughing_factor)
            
            xpoints = xpoints[::self.roughing_factor, ::self.roughing_factor]
            ypoints = ypoints[::self.roughing_factor, ::self.roughing_factor]
            zpoints = zpoints[::self.roughing_factor, ::self.roughing_factor]
        
        
        
        # Flip even rows so we cut in snake pattern
        xpoints[1::2,:] = np.flip( xpoints[1::2,:], axis=1)
        ypoints[1::2,:] = np.flip( ypoints[1::2,:], axis=1) # Unaffected, done for clarity
        zpoints[1::2,:] = np.flip( zpoints[1::2,:], axis=1)
        
        # Unravel
        xpoints = xpoints.ravel()
        ypoints = ypoints.ravel()
        zpoints = zpoints.ravel()
        return xpoints, ypoints, zpoints
    
    
    def compute_2d_rolling_maximum(self, arr, window):
        arr_padded = np.pad(arr, window, constant_values = arr.min())
    
        # Find maximum of elements in each row
        b = arr_padded
        for row in range(arr_padded.shape[0]):
            for _ in range(window):
                b[row,:] = np.maximum(np.roll(arr_padded[row,:], 1), b[row,:])
                b[row,:] = np.maximum(np.roll(arr_padded[row,:], -1), b[row,:])
    
        # Find maximum of elements in each column
        c = arr_padded
        for col in range(arr_padded.shape[1]):
            for _ in range(window):
                c[:,col] = np.maximum(np.roll(arr_padded[:,col], 1), c[:,col])
                c[:,col] = np.maximum(np.roll(arr_padded[:,col], -1), c[:,col])
    
        # Convert back to MODEL_ZS
        return np.maximum(b, c)[window:-window, window:-window]

