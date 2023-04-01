#%%
import numpy as np
import matplotlib.pyplot as plt
import utils_maps as um
import cv2 as cv
import os

folder = 'missionridge'
sw = {'lat':47.267153, 'lng':-120.4398904}
ne = {'lat':47.297119, 'lng':-120.391744}

sw, ne = um.snap_to_GPS_points(sw, ne)

# Get elevation data between gps coords

# Generate and save elevation data as grayscale image
elevations = um.get_elevation_data(sw, ne)
elevation_path = os.path.join(folder, 'elevations.png')
img = np.uint8(elevations/elevations.max()*255)
plt.imsave(elevation_path, img, cmap='gray')

# Generate and save map using tiled static map calls to Google API
tiled_map = um.generate_tiled_map(sw, ne, max_map_size=1200)
map_path = os.path.join(folder, 'googlemap.png')
cv.imwrite(map_path, tiled_map)
