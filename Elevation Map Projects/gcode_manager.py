# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 18:14:53 2023

@author: afisher
"""
import os
import numpy as np
import utils_maps as um
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter


class gcode_manager():
    def __init__(self, folder, model):
        self.folder = folder
        self.model = model
        self.parse_elevation_data()

        self.gcode = ''
        self.zsafe = 5
        self.feed_rate = 2000
        self.xsafe = -20
        self.roughing_factor = 4        
        return
    
    def parse_elevation_data(self):
        # Read elevations from file
        sw, ne = um.get_GPS_coords(self.folder)
        
        # Determine elevation and gradients
        self.elevations = um.get_elevation_data(sw, ne)
        row_gradients = (self.elevations[1:-1,2:] - self.elevations[1:-1,:-2])/2
        col_gradients = (self.elevations[2:,1:-1] - self.elevations[:-2,1:-1])/2
        self.elevation_gradients = np.pad(np.sqrt(row_gradients**2 + col_gradients**2), 1)
        
        # Convert to mm
        elev_H, elev_W = self.elevations.shape
        elevft_to_mm = self.model['T'] / np.ptp(self.elevations)
        elevpx_to_mm = self.model['H'] / (elev_H-1)

        self.elevft_to_mm = elevft_to_mm
        self.elevpx_to_mm = elevpx_to_mm
        self.z_heights = self.elevations * elevft_to_mm

        # Smooth gradients over a couple datapoints to avoid noise
        self.gradients = gaussian_filter(self.elevation_gradients * elevft_to_mm / elevpx_to_mm, 2)



        self.z_interp = RegularGridInterpolator((range(elev_H), range(elev_W)), self.elevations)
        self.gradient_interp = RegularGridInterpolator((range(elev_H), range(elev_W)), self.gradients)

        # Define width of model
        self.model['W'] = self.model['H'] / (ne['lat']-sw['lat'])* (ne['lng']-sw['lng']) 
        print('---- MODEL DIMENSIONS ----')
        print(f"X = {self.model['W']:.1f}mm, Y = {self.model['H']:.1f}mm, Z = {self.model['T']:.1f}mm")
        return


    def interpolate_elevation(self, xpoints, ypoints):
        # Convert points to elevation_indices
        elev_H, elev_W = self.elevations.shape
        xindices = xpoints / self.model['W'] * (elev_W-1)
        yindices = -ypoints / self.model['H'] * (elev_H-1)

        # Apply interpolations
        zpoints = self.z_interp((yindices, xindices))
        gradients = self.gradient_interp((yindices, xindices))

        return zpoints, gradients

    def save_gcode(self, file):
        path = os.path.join('Projects', self.folder, file)
        with open(path, 'w') as f:
            f.write(self.gcode)
        print(f'Saved "{file}"')
        
        
    def append_gcode_for_path(self, xpoints, ypoints, zpoints=None, roughing=False, lift=True, zdepth=0, is_contour=False, is_smoothing=False, tool_radius=2):
        # Compute zpoints if none_supplied
        if zpoints is None:
            zpoints, gradients = self.interpolate_elevation(xpoints, ypoints)

        # If contour, increase depth where gradient is nonzero
        if is_contour:
            zpoints += zdepth*np.sqrt(1+gradients**2)

        # If smoothing, increase height of tool so that every part of tool radius stays above surface
        if is_smoothing:
            zpoints += zdepth + tool_radius*gradients**2/np.sqrt(1+gradients**2)



        # Go to xy starting point
        self.gcode += f'G1 X{xpoints[0]:.2f} Y{ypoints[0]:.2f}\n'
        
        # If roughing, start off stock at safe xposition, and move to correct z depth
        if roughing:
            self.gcode += f'G1 X{self.xsafe:.2f} Y0\n'
            self.gcode += f'G1 Z{zpoints[0]:.2f}\n'
        
        # Move along points in path
        lines = [f'G1 X{x:.2f} Y{y:.2f} Z{z:.2f}\n' for x,y,z in
                 zip(xpoints, ypoints, zpoints)]
        self.gcode += ''.join(lines)
            
        # List to safe height
        if lift:
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
        
        unique_y = np.arange(0, -self.model['H'], -stepover)
        unique_x = np.arange(0, self.model['W'], stepover)
        [xpoints, ypoints] = np.meshgrid(unique_x, unique_y)
            
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
        byrow = arr_padded
        for row in range(arr_padded.shape[0]):
            for _ in range(window):
                byrow[row,:] = np.maximum(np.roll(arr_padded[row,:], 1), byrow[row,:])
                byrow[row,:] = np.maximum(np.roll(arr_padded[row,:], -1), byrow[row,:])
    
        # Find maximum of elements in each column
        bycol = arr_padded
        for col in range(arr_padded.shape[1]):
            for _ in range(window):
                bycol[:,col] = np.maximum(np.roll(arr_padded[:,col], 1), bycol[:,col])
                bycol[:,col] = np.maximum(np.roll(arr_padded[:,col], -1), bycol[:,col])
    
        # Convert back to model_zs
        return np.maximum(byrow, bycol)[window:-window, window:-window]

