#%%
import numpy as np
import matplotlib.pyplot as plt
import utils_maps as um


sw = {'lat':47.267153, 'lng':-120.4398904}
ne = {'lat':47.297119, 'lng':-120.391744}


um.generate_elevation_jpg(sw, ne, file='missionridge_elevations.png')
um.generate_tiled_map(sw, ne, file='missionridge_runs.png', max_map_size=2400)

# %%
