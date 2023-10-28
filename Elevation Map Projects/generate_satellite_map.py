#%%
import numpy as np
import matplotlib.pyplot as plt
import utils_maps as um
import cv2 as cv
import os



folder = 'altasnowbird'
sw, ne = um.get_GPS_coords(folder)


# Get satellite map from calls to Google Maps API
MAX_MAP_SIZE = 2400                                                             # max size of full map
WATERMARK_PIXELS = 60
MAX_TILE_WIDTH = 640                                                            # max size of single google API call
MAX_TILE_HEIGHT = MAX_TILE_WIDTH - 2*WATERMARK_PIXELS                           #Allow for cutting watermarked pixels
MAX_TILES = 26



# Determine latlngs for tiling
zoom, (H, W) = um.estimate_map_size(sw, ne, MAX_MAP_SIZE)

# Compute number of lat and lng tiles (have to remove Googlemaps watermark)
lat_tiles = H//MAX_TILE_HEIGHT + 1
lng_tiles = W//MAX_TILE_WIDTH + 1

if np.prod([lat_tiles, lng_tiles])>MAX_TILES:
    raise ValueError('Number of map tiles exceeds MAX_TILES')
    
    
# Compute equidistant lats and lngs
lats = np.linspace(sw['lat'], ne['lat'], lat_tiles+1)
lngs = np.linspace(sw['lng'], ne['lng'], lng_tiles+1)

# Download tiles and append
for jlat in range(lat_tiles):
    for jlng in range(lng_tiles):
        
        # Get GPS of tile, 
        # To match image coord system, work from largest to smallest lattitude.
        tile_sw = {'lat':lats[-(jlat+2)], 'lng':lngs[jlng+0]}
        tile_ne = {'lat':lats[-(jlat+1)], 'lng':lngs[jlng+1]}

        # Download tile to file and read
        tile_H, tile_W = um.download_tile(tile_sw, tile_ne, zoom, WATERMARK_PIXELS)
        tile_img = cv.imread('tile.png', cv.IMREAD_GRAYSCALE)
        
        # Initialize fullsize image once tile image dimensions are known.
        if jlat==0 and jlng==0:
            full_img = np.zeros((lat_tiles*tile_H, lng_tiles*tile_W), dtype=np.uint8)
        
        # Remove watermark and add to full image
        full_img[tile_H*jlat:tile_H*(jlat+1), tile_W*jlng:tile_W*(jlng+1)] = tile_img[WATERMARK_PIXELS:-WATERMARK_PIXELS,:]
os.remove('tile.png')

# Resize image to remove mercantor projection effects
center_lat = sw['lat']/2+ne['lat']/2
vert_scale = np.cos( center_lat *np.pi/180)
new_shape = full_img.shape[1], int(full_img.shape[0]*vert_scale)
resized_img = cv.resize(full_img, new_shape)

# # Save image to file
map_path = os.path.join('Projects', folder, 'satellite.png')
cv.imwrite(map_path, resized_img)




# %%
