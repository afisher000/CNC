# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 15:29:53 2023

@author: afisher
"""

# %% OLD CODE
# elevation_path = os.path.join(folder, 'elevations.png')
# img = np.uint8(elevations/elevations.max()*255)
# plt.imsave(elevation_path, img, cmap='gray')

# Generate and save elevation data as grayscale image
elevations = um.get_elevation_data(sw, ne)