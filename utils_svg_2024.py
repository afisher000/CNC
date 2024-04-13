# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 18:33:53 2023

@author: afisher
"""

# %%
import xml.etree.ElementTree as ET
import os
import numpy as np

def get_svg_paths(svg_path, dimensions):    
    # Get paths in svg file
    tree = ET.parse(svg_path)
    root = tree.getroot()
    paths = root.findall('.//{http://www.w3.org/2000/svg}path')  
    
    # Get scale factor
    height = float(root.get('height').strip('mm'))
    scale = dimensions['height']/height
    

    # Write definitions in dictionary immediately? Avoid cluttering namespace. 
    # Handler
    def m_command(j, commands):
        return j+1, commands[j]
    def l_command(j, commands):
        return j+1, commands[j]
    def c_command(j, commands):
        return j+3, commands[j+2]
    def v_command(j, commands):
        return j+1, commands[j]
    def h_command(j, commands):
        return j+1, commands[j]
    
    command_handler = {
        'm':m_command,
        'l':l_command,
        'c':c_command,
        'v':v_command,
        'h':h_command
    }
    
    contours = []
    for path in paths:
        # Get path styles
        styles = {}
        if path.get('style') is not None:
            for pair in path.get('style').split(';'):
                key, value = pair.split(':')
                styles[key] = value

            stroke_width = float(styles['stroke-width'])
            styles['stroke-width'] = stroke_width
            dasharray = styles['stroke-dasharray']
            if dasharray=='none':
                styles['stroke-dasharray'] = None
            else:
                styles['stroke-dasharray'] = (np.array(list(map(float, styles['stroke-dasharray'].split(','))))/stroke_width).astype(int)

        # Get path commands
        d = path.get('d')
        commands = d.split()   
    
        # Initialize
        cur_command = None
        curx = 0
        cury = 0
        contour = []
        j = 0
        while j<len(commands):
            command = commands[j]
            if command.lower() in command_handler.keys():
                cur_command = command
                j, pointstr = command_handler[command.lower()](j+1, commands)
            else:
                j, pointstr = command_handler[cur_command.lower()](j, commands)
            
            # Lowercase is relative movement, uppercase is absolute
            try:
                x,y = map(float, pointstr.split(','))
            except:
                if cur_command.lower()=='v':
                    x = 0
                    y = float(pointstr)
                elif cur_command.lower()=='h':
                    x = float(pointstr)
                    y = 0
                else:
                    raise ValueError(pointstr)
            if cur_command.islower():
                curx += x
                cury += y
            else:
                curx = x
                cury = y
                    
            contour.append((curx,cury))
            
        # Save to contours, ensure same dimensions as model
        contours.append([np.array(contour) * scale, styles])

    return optimize_contour_order(contours)

def optimize_contour_order(cs):
    # Each contour is 1x2 list: [xypoints, styles]
    # Get endpoints
    starts      = np.array([c[0][0] for c in cs])
    ends        = np.array([c[0][-1] for c in cs])

    # Create set to keep track of indices 
    ordered_cs  = []
    idxs        = set(range(len(cs)))

    # Start with first contour
    idx         = 0
    idxs.remove(0)
    ordered_cs.append(cs[idx])
    cur         = cs[idx][0][-1]
    

    while idxs:
        mask        = list(idxs)
        start_dists = np.linalg.norm(starts[mask] - cur, axis=1)
        end_dists   = np.linalg.norm(ends[mask] - cur, axis=1)

        if min(start_dists)<min(end_dists):
            idx     = mask[np.argmin(start_dists)]
            idxs.remove(idx)
            ordered_cs.append(cs[idx])
            cur     = cs[idx][0][-1]
        else:
            idx     = mask[np.argmin(end_dists)]
            idxs.remove(idx)
            ordered_cs.append([cs[idx][0][::-1], cs[idx][1]])
            cur     = cs[idx][0][0]
    return ordered_cs
            


def optimize_contour_order_old(contours):
    ## CAN THIS BE IMPROVED??
    # Order contours to reduce travel time
    start_points = np.array([contour[0] for contour in contours])
    end_points = np.array([contour[-1] for contour in contours])
    
    # Start with first contour
    idxs = list(range(len(contours))) # keeps track of which idxs still need cut
    del idxs[0]
    contour = contours[0]
    ordered_contours = [contour] # keeps track of the ordered contours
    current_point = contour[-1]
    
    # Iterate over remaining contours
    while len(idxs)>0:
        # Find shortest path to next contour (either end)
        start_dists = np.linalg.norm(start_points[idxs] - current_point, axis=1)
        end_dists = np.linalg.norm(end_points[idxs] - current_point, axis=1)  
        
        if min(start_dists)<min(end_dists):
            idx = np.argmin(start_dists)
            contour = contours[idxs[idx]]
        else:
            idx = np.argmin(end_dists)
            contour = contours[idxs[idx]][::-1]
            
        
        # Remove from list of idxs and append. Update current point
        del idxs[idx]
        ordered_contours.append(contour)
        current_point = contour[-1]
    return ordered_contours

def add_points_to_path(path, max_distance):
    # Unpack path
    xs = path[:,0]
    ys = path[:,1]
    
    # Calculate distances between successive points along path
    distances = np.sqrt( np.diff(xs)**2 + np.diff(ys)**2)
    
    # Calculate number of interpolating points for each line in path (minimum of 2)
    npoints = np.ceil(distances/max_distance+1).astype(int)
    
    # Loop over moves in path, extend with new discretized paths
    new_xs = []
    new_ys = []
    for j in range(len(distances)):
        new_xs.extend( np.linspace(xs[j], xs[j+1], npoints[j]) )
        new_ys.extend( np.linspace(ys[j], ys[j+1], npoints[j]) )

    return np.array(new_xs), np.array(new_ys)
    
# %%
