# %%
import os
from pprint import pprint
import requests
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

GLOBE_SIZE = 256
MAX_MAP_SIZE = 640
WATERMARK_PIXELS = 60
MAX_TILING_MAPS = 17
API_KEY = os.environ['MAPS_API_KEY']

def get_elevation_data(sw, ne):
    NPOINTS=3601
    
    # Check both gps coords are in same geographic tile. Generalize in next update.
    if int(sw['lat']//1)==int(ne['lat']//1) and int(sw['lng']//1)==int(ne['lng']//1):
        hgtLat, hgtLng = int(sw['lat']//1), int(sw['lng']//1)
    else:
        raise ValueError('GPS coords are on different geographic tiles. Will be allowed in next release')
    
    # Create hgt_file string
    latCard = 'N' if hgtLat>=0 else 'S'
    lngCard = 'E' if hgtLng>=0 else 'W'
    hgt_path = os.path.join('HGT Files', f'{latCard}{abs(hgtLat)}{lngCard}{abs(hgtLng)}.hgt')
    
    # Read the binary elevation data from the SRTM HGT file
    with open(hgt_path, 'rb') as f:
        data = f.read()
    elevation_data = np.frombuffer(data, dtype='>i2').reshape((NPOINTS, NPOINTS))
    
    # Get elevations between sw, ne gps points
    # Index counts down and right from top left corner
    # min latitude is at idx=NPOINTS and max latitude is at idx=0
    # min longitude is at idx=0 and max longitude is at idx=NPOINTS
    def getLatIdx(lat): return NPOINTS-int((lat%1)*NPOINTS)
    def getLngIdx(lng): return int((lng%1)*NPOINTS)
    
    jlatmin, jlatmax = getLatIdx(ne['lat']), getLatIdx(sw['lat'])
    jlngmin, jlngmax = getLngIdx(sw['lng']), getLngIdx(ne['lng'])
    elevations = elevation_data[jlatmin:jlatmax, jlngmin:jlngmax]
    return elevations

def estimate_zoom_value(frac, MAX_MAP_SIZE):
    return int(np.log(MAX_MAP_SIZE/GLOBE_SIZE/frac) / np.log(2))

def estimate_pixel_value(zoom, fraction):
    return int(2**zoom * GLOBE_SIZE * fraction)

def estimate_map_size(sw, ne, max_map_size=MAX_MAP_SIZE):
    # Google maps uses mercantor projection which changes spacing between 
    # lines of latitude. Extra math is needed to calculate latitude fraction.
    def latRad(lat):
        sin = np.sin(lat*np.pi/180)
        rad = np.log((1+sin)/(1-sin))/2
        return rad/2
    
    # Compute fractions
    lat_fraction = (latRad(ne['lat']) - latRad(sw['lat']))/np.pi
    lng_fraction = (ne['lng'] - sw['lng'])/360

    # Estimate zooms
    lat_zoom = estimate_zoom_value(lat_fraction, max_map_size)
    lng_zoom = estimate_zoom_value(lng_fraction, max_map_size)
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
        "size": f'{lng_px}x{lat_px+2*WATERMARK_PIXELS}',
        "maptype": "hybrid",
        "key": API_KEY,
        "style": styles
    }
    response = requests.get(url, params=params)
    # print(f'Reading at zoom {zoom}')
    with open('tile.png', 'wb') as f:
        f.write(response.content)
    return (lat_px, lng_px)

def generate_tiled_map(sw, ne, file='tiled_map.jpg', max_map_size=1200):
    styles = {
        "feature:all|element:all|visibility:off", #clear all
        "feature:poi.sports_complex|element:all|visibility:on",
        "feature:poi.sports_complex|element:geometry|visibility:on|color:0xffffff" #turn ski run white
    }
    # styles = {}
    # Single download for testing
    # download_map(sw, ne, styles=styles)

    # Determine latlngs for tiling
    zoom, (lat_px, lng_px) = estimate_map_size(sw, ne, max_map_size=max_map_size)

    tile_shape = (lat_px//(MAX_MAP_SIZE-2*WATERMARK_PIXELS)+1, lng_px//MAX_MAP_SIZE+1)
    lats = np.linspace(sw['lat'], ne['lat'], tile_shape[0]+1)
    lngs = np.linspace(sw['lng'], ne['lng'], tile_shape[1]+1)
    # print(f'Tile shape = {tile_shape}')
    # print(f'Overall size = {lat_px, lng_px}')

    if np.prod(tile_shape)>MAX_TILING_MAPS:
        raise ValueError(f'Number of maps for tiling ({np.prod(tile_shape)}) exceeds MAX_TILING_MAPS ({MAX_TILING_MAPS})')

    else:
        # Download images
        for jlat in range(len(lats)-1):
            for jlng in range(len(lngs)-1):
                # Compute tile latlngs and get download map
                sw = {'lat':lats[-jlat-2], 'lng':lngs[jlng]}
                ne = {'lat':lats[-jlat-1], 'lng':lngs[jlng+1]}
                h,w = download_map(sw, ne, styles=styles)
                img = cv.imread('tile.png', cv.IMREAD_GRAYSCALE)

                # Initialize image
                if jlat==0 and jlng==0:
                    tiled_img = np.zeros((tile_shape[0]*h, tile_shape[1]*w), dtype=np.uint8)

                # Add tile to image
                tiled_img[h*jlat:h*(jlat+1), w*jlng:w*(jlng+1)] = img[WATERMARK_PIXELS:-WATERMARK_PIXELS,:]

        # Undo mercantor projection for compatibility with elevation map
        lat = sw['lat']/2+ne['lat']/2
        vert_scale = np.cos(lat*np.pi/180)
        height, width = int(tiled_img.shape[0]*vert_scale), tiled_img.shape[1]
        print(height, width)
        resized_tiled_img = cv.resize(tiled_img, (width, height))
        
        # Save as color image to avoid issues detecting gray later
        color_img = cv.cvtColor(resized_tiled_img, cv.COLOR_GRAY2BGR)
        color_img[:,:,0] += 1
        cv.imwrite(file, color_img)
        print(f'Saved tiled google map as {file}')
        os.remove('tile.png')
    return



# %%
