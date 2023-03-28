#%%
import numpy as np
import matplotlib.pyplot as plt
import utils_maps as um
import cv2 as cv
import os

sw = {'lat':47.267153, 'lng':-120.4398904}
ne = {'lat':47.297119, 'lng':-120.391744}

# Get elevation data between gps coords
elevations = um.get_elevation_data(sw, ne)

# Save elevations to grayscale image
file = 'missionridge_elevations.png'
img = np.uint8(elevations/elevations.max()*255)
cv.imwrite(file, img)
plt.imsave(file, img, cmap='gray')
print(f'Saved elevation image as {file}')

# Generate map using tiled static map calls to Google API
# um.generate_tiled_map(sw, ne, file='test.png', max_map_size=2400)



