# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 18:33:53 2023

@author: afisher
"""

# %%
import xml.etree.ElementTree as ET
import os
import numpy as np
# %%
def get_svg_paths(folder, MODEL):
    svg_path = os.path.join('Projects', folder, 'contours.svg')
    
    # Get paths in svg file
    tree = ET.parse(svg_path)
    root = tree.getroot()
    paths = root.findall('.//{http://www.w3.org/2000/svg}path')  
    
    height = float(root.get('height').strip('mm'))
    
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
            
        # Save to contours, ensure same dimensions as MODEL
        contours.append(np.array(contour) * MODEL['H']/height)

        ordered_contours = optimize_contour_order(contours)
    return ordered_contours


def optimize_contour_order(contours):
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
