# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 19:47:47 2023

@author: afisher
"""

import cv2 as cv
import numpy as np
import utils_contours as uc
import utils_smoothing as us

file = 'maxwell'
gray = cv.imread(file+'.jpg', cv.IMREAD_GRAYSCALE)
ret, img = cv.threshold(gray, 230, 255, cv.THRESH_BINARY)
white = np.full_like(img, 255, dtype=np.uint8)

# Find contours, remove small/large areas
all_contours, hierarchy = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
areas = np.array([cv.contourArea(c) for c in all_contours])
minArea, maxArea = 40, 1e5
contours = np.delete(
    np.array(all_contours, dtype=object), 
    np.where((areas<minArea)|(areas>maxArea))[0],
    0
)

cv.drawContours(white, contours, -1, 0, 2)
# uc.showImage(white)
uc.save2SVG(file+'.svg', contours, white.shape)