#%%
import numpy as np
import matplotlib.pyplot as plt
import utils_maps as um
import cv2 as cv
import os

# folder = 'missionridge'
# sw = {'lat':47.267153, 'lng':-120.4398904}
# ne = {'lat':47.297119, 'lng':-120.391744}

folder = 'simple_mammoth'
sw = {'lat':37.618210, 'lng':-119.060685}
ne = {'lat':37.656038, 'lng':-119.0001}

# folder = 'mammoth'
# sw = {'lat':37.618210, 'lng':-119.060685}
# ne = {'lat':37.656038, 'lng':-119.0001}

# folder = 'bigsky'
# sw = {'lat':45.253139, 'lng':-111.477485}
# ne = {'lat':45.308380, 'lng':-111.358708}

# Snap gps coords to nearest elevation data points
sw, ne = um.snap_to_GPS_points(sw, ne)

# Generate and save elevation data as grayscale image
elevations = um.get_elevation_data(sw, ne)
elevation_path = os.path.join(folder, 'elevations.png')
img = np.uint8(elevations/elevations.max()*255)
plt.imsave(elevation_path, img, cmap='gray')

# Generate and save map using tiled static map calls to Google API
tiled_map = um.generate_tiled_map(sw, ne, max_map_size=3600)
map_path = os.path.join(folder, 'googlemap.png')
cv.imwrite(map_path, tiled_map)

# %%
