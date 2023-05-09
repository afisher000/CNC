# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 18:14:53 2023

@author: afisher
"""
import os
import numpy as np
import utils_maps as um
import utils_contours as uc
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter
import xml.etree.ElementTree as ET


class gcode_manager():
    def __init__(self, folder, model):
        self.folder = folder
        self.model = model
        self.parse_elevation_data()

        self.gcode = ''
        self.time = 0
        self.zsafe = 5
        self.feed_rate = 2000
        self.xsafe = -20
        self.roughing_grid_points = 5      
        return
    
    def parse_elevation_data(self):
        # Read elevations from file
        sw, ne = um.get_GPS_coords(self.folder)
        
        # Define width of model
        self.model['W'] = self.model['H'] / (ne['lat']-sw['lat'])* (ne['lng']-sw['lng']) 
        print('---- MODEL DIMENSIONS ----')
        print(f"X = {self.model['W']:.1f}mm, Y = {self.model['H']:.1f}mm, Z = {self.model['T']:.1f}mm")
        
        # Determine elevation and gradients
        self.elevations = um.get_elevation_data(sw, ne)
        row_gradients = (self.elevations[1:-1,2:] - self.elevations[1:-1,:-2])/2
        col_gradients = (self.elevations[2:,1:-1] - self.elevations[:-2,1:-1])/2
        self.elevation_gradients = np.pad(np.sqrt(row_gradients**2 + col_gradients**2), 1)
        
        # Convert to mm
        elev_H, elev_W = self.elevations.shape
        elevft_to_mm = self.model['T'] / np.ptp(self.elevations)
        elevpx_to_mm = self.model['H'] / (elev_H-1)
        self.z_heights = (self.elevations-self.elevations.max()) * elevft_to_mm

        # Smooth gradients over a couple datapoints to avoid noise
        self.gradients = gaussian_filter(self.elevation_gradients * elevft_to_mm / elevpx_to_mm, 2)

        # Define interpolations to map x,y [mm] to z [mm] or gradient [mm/mm]
        ygrid_points = -1*np.array(range(elev_H)) * self.model['H'] / (elev_H-1)
        xgrid_points = np.array(range(elev_W)) * self.model['W'] / (elev_W-1)

        # Have to use np.flip so grid vectors are increasing
        self.z_interp = RegularGridInterpolator((np.flip(ygrid_points), xgrid_points), np.flip(self.z_heights, axis=0))
        self.gradient_interp = RegularGridInterpolator((np.flip(ygrid_points), xgrid_points), np.flip(self.gradients, axis=0))
        

        return


    def interpolate_points(self, xpoints, ypoints):
        # Clip avoid model boundaries
        eps = 1e-6
        clipped_xpoints = np.clip(xpoints, eps, self.model['W']-eps)
        clipped_ypoints = np.clip(ypoints, -self.model['H']+eps, eps)


        # Apply interpolations
        zpoints = self.z_interp((clipped_ypoints, clipped_xpoints))
        gradients = self.gradient_interp((clipped_ypoints, clipped_xpoints))

        return zpoints, gradients


    def generate_roughing_gcode(self, file, 
            tool_diameter=0.25*25.4, stepover_frac=0.8, max_stepdown=None):
        
        # Compute stepover and number of stepdowns
        stepover = tool_diameter*stepover_frac
        if max_stepdown is None:
            n_stepdowns = 1
        else:
            n_stepdowns = int(np.ceil(self.model['T'] / max_stepdown))
            
        # Compute points in raster scan
        xpoints, ypoints = self.get_raster_points(stepover)
        zpoints = self.get_roughing_heights(xpoints, ypoints, tool_diameter)
        
        # Loop over raster scans to generate gcode
        self.initialize_gcode()
        for j in range(n_stepdowns):
            self.add_path_to_gcode(xpoints, ypoints, zpoints*(j+1)/n_stepdowns, start_safe=True)
            
            # Finish raster at safe height
            self.lift()
        self.save_gcode(file)
        return
        
    def generate_smoothing_gcode(self, file,
            tool_diameter=4, stepover=1):
        
        # Compute points and gradients
        xpoints, ypoints = self.get_raster_points(stepover)
        zpoints, gradients = self.interpolate_points(xpoints, ypoints)
        
        # Correct zpoints with gradient information
        radius = tool_diameter/2
        zpoints += radius*gradients/np.sqrt(1+gradients**2)
        
        # Generate gcode for smoothing
        self.initialize_gcode()
        self.add_path_to_gcode(xpoints, ypoints, zpoints)
        self.lift()
        self.save_gcode(file)
        return
        

    
    
    def generate_runs_gcode(self, file,
            max_distance=0.5, transition_height=5, zdepth=-1):
        
        paths = uc.get_svg_paths(self.folder, self.model)
        
        
        # Loop over paths
        self.initialize_gcode()
        for j in range(len(paths)):
            path = paths[j]
            
            # Add points to contours to satisfy max_distance
            xpoints, ypoints = self.add_points_to_path(path, max_distance)
            zpoints, gradients = self.interpolate_points(xpoints, ypoints)
            
            # Correct zpoints with gradient information and append to gcode
            zpoints += zdepth*np.sqrt(1+gradients**2)
            self.add_path_to_gcode(xpoints, ypoints, zpoints)
        
            # Add transition between end of current path and beginning of next path
            if j<(len(paths)-1):
                transition = np.vstack([paths[j][-1], paths[j+1][0]])
                xpoints, ypoints = self.add_points_to_path(transition, max_distance)
                zpoints, gradients = self.interpolate_points(xpoints, ypoints)
                
                # Add gcode for transition, bit will be at transition_height.
                self.add_path_to_gcode(xpoints, ypoints, zpoints+transition_height)
        
        self.lift()
        self.save_gcode(file)
        return
    
    def read_svg_image_properties(self, image_id):
        # Load the SVG file
        svg_path = os.path.join('Projects', self.folder, 'contours.svg')
        tree = ET.parse(svg_path)
        root = tree.getroot()
        
        # Find the raster image elements and extract its attributes
        image_elements = root.findall(".//{http://www.w3.org/2000/svg}image")
        for image_element in image_elements:
            if image_element.get('id') == image_id:
                x = float(image_element.get('x'))
                y = float(image_element.get('y'))
                width = float(image_element.get('width'))
                height = float(image_element.get('height'))
                transform = image_element.get('transform')
                
                # Correct inversion
                if transform=='scale(-1)': 
                    x = -1*(x+width)
                    y = -1*(y+height)
                    
                # Change sign on y to match cnc coordinates
                return x, -y, width, height, transform
        
        raise ValueError(f'No image with the id {image_id} in contours.svg')
    
    
    def generate_image_gcode(self, image_id,
            transition_height=5):
        
        # Read paths and boundaries from fusion logo
        paths = self.parse_fusion_gcode(image_id+'.nc')
        
        # Reads properties of image in contour.svg
        x, y, width, height, transform = self.read_svg_image_properties(image_id)
        
        # Loop over paths
        self.initialize_gcode()
        for path in paths:
            # Get xypoints and depth from fusion path
            xpoints, ypoints, zdepth = zip(*path)
            
            # Apply inversion correction
            if transform=='scale(-1)':
                xpoints = width - xpoints
                ypoints = - height - ypoints
                
            # Get elevation interpolation
            zpoints, gradients = self.interpolate_points(xpoints, ypoints)
            
            # Add path to gcode, lift for transition
            self.add_path_to_gcode(x + xpoints, y + ypoints, zdepth + zpoints)
            self.lift(zpoints[-1]+transition_height)
        
        # Lift to safety and save gcode
        self.lift()
        self.save_gcode(image_id + '.txt')
        return
        
    
    def parse_fusion_gcode(self, file):
        with open(file, 'r') as f:
            lines = f.readlines()
        
        # Initialize position (Z=5 so no cutting by default)
        position = {'X':0, 'Y':0, 'Z':5}
        
        # Create list for holding paths
        paths, path = [], []
        
        # Separate line into commands
        for line in lines:
            is_new_position = False
            commands = line.strip('\n').split(' ')
            for command in commands:
                # Skip blank lines or comments starting with "("
                if len(command)==0 or command[0]=='(':
                    # print(f'Command = "{command}" skipped')
                    break #Break out of commands loop
                    
                # If X,Y, or Z command, update position
                elif command[0] in position.keys():
                    position[command[0]] = float(command[1:])
                    is_new_position = True
        
            # If z is cutting at a new position, add to path and update bounds
            if position['Z']<0 and is_new_position:
                path.append([position['X'], position['Y'], position['Z']])

                
            # If z is not cutting, add to paths (if nonempty) and reset path
            elif position['Z']>0:
                if len(path)>2:
                    paths.append(path)
                path = []
        return paths
    

        
        
        
        
        
    def add_points_to_path(self, path, max_distance):
        # Unpack path (change yhat direction from png)
        xpoints = path[:,0]
        ypoints = -1*path[:,1]
        
        # Calculate distances between successive points
        xdistances = np.diff(xpoints)
        ydistances = np.diff(ypoints)
        distances = np.sqrt(xdistances**2+ydistances**2)
        
        # Calculate points per move (minimum of 2)
        points_per_move = np.ceil(distances/max_distance+1).astype(int)
        
        # Define containers for new points
        new_xpoints = []
        new_ypoints = []
        
        # Loop over moves in path, extend with new discretized paths
        for j in range(len(distances)):
            new_xpoints.extend( np.linspace(xpoints[j], xpoints[j+1], points_per_move[j]) )
            new_ypoints.extend( np.linspace(ypoints[j], ypoints[j+1], points_per_move[j]) )

        return np.array(new_xpoints), np.array(new_ypoints)
        
        
    def add_path_to_gcode(self, xpoints, ypoints, zpoints, start_safe=False):
        # Go to starting points
        if start_safe:
            self.gcode += f'G1 X{self.xsafe:.2f} Y0\n'
            self.gcode += f'G1 Z{zpoints[0]:.2f}\n'
        else:
            self.gcode += f'G1 X{xpoints[0]:.2f} Y{ypoints[0]:.2f}\n'
        
        # Move along points in path
        lines = [f'G1 X{x:.2f} Y{y:.2f} Z{z:.2f}\n' for x,y,z in
                 zip(xpoints, ypoints, zpoints)]
        self.gcode += ''.join(lines)
        
        # Add time for path
        distances = np.sqrt(np.diff(xpoints)**2+np.diff(ypoints)**2+np.diff(zpoints)**2)
        self.time += distances.sum()/self.feed_rate
        return

    def lift(self, z=None):
        if z is None:
            self.gcode += f'G1 Z{self.zsafe:.2f}\n' 
        else:
            self.gcode += f'G1 Z{z:.2f}\n' 
        return
    
    def get_roughing_heights(self, xpoints, ypoints, tool_diameter):
        # Initialize an array of zpoints
        zpoints_max = np.full_like(xpoints, fill_value=-self.model['T'])
        
        # Loop over offsets within tool radius
        xy_offset = np.linspace(-tool_diameter/2, tool_diameter/2, self.roughing_grid_points)
        for dx in xy_offset:
            for dy in xy_offset:
                # Clip to keep within model boundaries
                clipped_xpoints = np.clip(xpoints+dx, 0, self.model['W'])
                clipped_ypoints = np.clip(ypoints+dy, -self.model['H'], 0)
                
                # Update max zpoints
                zpoints, _ = self.interpolate_points(clipped_xpoints,clipped_ypoints)
                zpoints_max = np.maximum(zpoints_max, zpoints)
                
        return zpoints_max
        
    
    def initialize_gcode(self):
        # Initialize gcode (used fusion example as template)
        # G17 -> select xy plane, G21 -> metric units, G90 -> absolute moves
        self.gcode = f'G90 G94\nG17\nG21\nG90\nG54\nG1 Z{self.zsafe} F{self.feed_rate}\n'
        self.time = 0
    
    def save_gcode(self, file):
        gcode_folder = os.path.join('Projects', self.folder, 'Gcode')
        if not os.path.exists(gcode_folder):
            os.makedirs(gcode_folder)
            
        file_path = os.path.join(gcode_folder, file)
        with open(file_path, 'w') as f:
            f.write(self.gcode)
        print(f'Saved "{file}", estimate time = {self.time:.1f} min')
        return
    
    def get_raster_points(self, stepover):
        unique_y = np.arange(0, -self.model['H'], -stepover)
        unique_x = np.arange(0, self.model['W'], stepover)
        [xpoints, ypoints] = np.meshgrid(unique_x, unique_y)        
        
        # Flip X values in even rows so we cut in snake pattern
        xpoints[1::2,:] = np.flip( xpoints[1::2,:], axis=1)
        
        # Unravel
        xpoints = xpoints.ravel()
        ypoints = ypoints.ravel()
        return xpoints, ypoints
    


## OLD CODE
        # self.z_interp = RegularGridInterpolator((range(elev_H), range(elev_W)), self.z_heights)
        # self.gradient_interp = RegularGridInterpolator((range(elev_H), range(elev_W)), self.gradients)
        # # Convert points to elevation_indices
        # elev_H, elev_W = self.elevations.shape
        # xindices = xpoints / self.model['W'] * (elev_W-1)
        # yindices = -ypoints / self.model['H'] * (elev_H-1)