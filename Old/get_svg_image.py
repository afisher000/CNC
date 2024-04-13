# -*- coding: utf-8 -*-
"""
Created on Mon May  8 15:42:08 2023

@author: afisher
"""

import xml.etree.ElementTree as ET
from PIL import Image
import os

# # TODO
# Be able to handle rotation and scaling transformations



# Load the SVG file
svg_path = os.path.join('Projects', 'mammoth', 'contours.svg')
tree = ET.parse(svg_path)
root = tree.getroot()

# Find the raster image element and extract its attributes
image_elements = root.findall(".//{http://www.w3.org/2000/svg}image")  # adjust the selector based on your SVG file
for image_element in image_elements:
    # print(image_element.get("{http://www.w3.org/1999/xlink}href"))
    x = float(image_element.get('x'))
    y = float(image_element.get('y'))
    width = float(image_element.get('width'))
    height = float(image_element.get('height'))
    transform = image_element.get('transform')
    
    print(image_element.get('id'))
    
    #
