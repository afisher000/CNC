# %%
import os
import googlemaps
from pprint import pprint
import requests
import numpy as np
import math
import cv2 as cv

## TODO
# Get map defined by latlong limits, not center and zoom
# Apply styles


request_API = True
GLOBE_SIZE = 256
MAX_MAP_SIZE = 640


def estimate_zoom_value(frac, MAX_MAP_SIZE):
    return int(np.log(MAX_MAP_SIZE/GLOBE_SIZE/frac) / np.log(2))

def estimate_pixel_value(zoom, fraction):
    return int(2**zoom * GLOBE_SIZE * fraction)

def estimate_map_size(sw, ne, MAX_MAP_SIZE=MAX_MAP_SIZE):
    # Compute fractions
    lat_fraction = (ne['lat'] - sw['lat'])/180
    lng_fraction = (ne['lng'] - sw['lng'])/360

    # Estimate zooms
    lat_zoom = estimate_zoom_value(lat_fraction, MAX_MAP_SIZE)
    lng_zoom = estimate_zoom_value(lng_fraction, MAX_MAP_SIZE)
    zoom = min(lat_zoom, lng_zoom)

    # Construct size string
    lat_px = estimate_pixel_value(zoom, lat_fraction)
    lng_px = estimate_pixel_value(zoom, lng_fraction)

    return zoom, (lat_px, lng_px)


def download_map(sw, ne, styles={}):
    # Generate API call for a static map
    url = "https://maps.googleapis.com/maps/api/staticmap"
    zoom, (lat_px, lng_px) = estimate_map_size(sw, ne)
    lat_center = sw['lat']/2 + ne['lat']/2
    lng_center = sw['lng']/2 + ne['lng']/2
    center = f'{lat_center},{lng_center}'

    params = {
        "center": center,
        "zoom": zoom,
        "size": f'{lat_px}x{lng_px}',
        "maptype": "roadmap",
        "key": API_KEY,
        "style": styles
    }
    response = requests.get(url, params=params)
    with open('tile.png', 'wb') as f:
        f.write(response.content)
    return (lat_px, lng_px)


# Load API key from environment variables
API_KEY = os.environ['MAPS_API_KEY']

sw = {'lat':47.267153, 'lng':-120.4398904}
ne = {'lat':47.297119, 'lng':-120.391744}

# WHY ARE MY DIMENSIONS WRONG?

styles = {
    "feature:all|element:all|visibility:off", #clear all
    "feature:poi.sports_complex|element:geometry|visibility:on|color:0xff0000" #turn on ski runs to red
}


# download_map(sw, ne, styles=styles)

# Determine latlngs for tiling
zoom, (lat_px, lng_px) = estimate_map_size(sw, ne, MAX_MAP_SIZE=1000)
tile_shape = (lat_px//MAX_MAP_SIZE+1, lng_px//MAX_MAP_SIZE+1)
lats = np.linspace(sw['lat'], ne['lat'], tile_shape[0]+1)
lngs = np.linspace(sw['lng'], ne['lng'], tile_shape[1]+1)
print(f'Tile shape = {tile_shape}')

# imgs = []
# # Download images
# for jlat in range(len(lats)-1):
#     for jlng in range(len(lngs)-1):
#         sw = {'lat':lats[jlat], 'lng':lngs[jlng]}
#         ne = {'lat':lats[jlat+1], 'lng':lngs[jlng+1]}
#         h,w = download_map(sw, ne, styles=styles)
#         img = cv.imread('tile.png', cv.IMREAD_GRAYSCALE)

#         # Initialize image 
#         if jlat==0 and jlng==0:
#             tiled_img = np.zeros((tile_shape[0]*h, tile_shape[1]*w), dtype=np.uint8)

#         #
#         tiled_img[h*jlat:h*(jlat+1), w*jlng:w*(jlng+1)] = img

# cv.imwrite('test_tiling.jpg', tiled_img)



# %%
