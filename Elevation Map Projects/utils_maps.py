# %%
import os
from pprint import pprint
import requests
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import RegularGridInterpolator


def get_GPS_coords(folder):
    '''Reads GPS coords from file. Snaps to nearest latlng data points in data
    folders.'''
    NPOINTS = 3601
    
    # Read data, round to nearest arcsecond (1/3600 of degree)
    GPS_df = pd.read_csv('GPS coordinates.csv', index_col=0)
    GPS_coords = GPS_df.loc[folder].apply(lambda x: (x//1) + int((x%1)*NPOINTS)/NPOINTS)
    
    # Write to sw and ne dictionaries
    sw, ne = {}, {}
    sw['lat'], sw['lng'], ne['lat'], ne['lng'] = GPS_coords
    return sw, ne


def zinterpolation(xpoints, ypoints, MODEL, elevations):
    # Define interpolation
    elev_H, elev_W = elevations.shape
    elev_interp = RegularGridInterpolator((range(elev_H), range(elev_W)), elevations)
    
    # Convert points to elevation_indices
    xindices = xpoints / MODEL['W'] * (elev_W-1)
    yindices = -ypoints / MODEL['H'] * (elev_H-1)
    elevation_values = elev_interp((yindices, xindices))

    
    # Evaluate interpolation and convert to mm
    zpoints = (elevation_values - elevations.max()) / np.ptp(elevations) * MODEL['T']
    return zpoints


def get_elevation_data(sw, ne):
    # Index counts down and right from top left corner
    # min latitude is at idx=NPOINTS and max latitude is at idx=0
    # min longitude is at idx=0 and max longitude is at idx=NPOINTS
    
    NPOINTS = 3601
    hgt_folder = 'HGT Files'
    
    # Round to lower integers
    hgtLats = range(int(sw['lat']//1), int(ne['lat']//1)+1)
    hgtLngs = range(int(sw['lng']//1), int(ne['lng']//1)+1)
    
    # Initialize elevation_data according to number of tiles
    all_elevations = np.zeros((
        (NPOINTS-1)*len(hgtLats)+1, 
        (NPOINTS-1)*len(hgtLngs)+1
    ))
    
    # Loop over hgt tiles
    for jLat, hgtLat in enumerate(hgtLats):
        for jLng, hgtLng in enumerate(hgtLngs):
    
            # Create hgt_file string
            latCard = 'N' if hgtLat>=0 else 'S'
            lngCard = 'E' if hgtLng>=0 else 'W'
            hgt_file = f'{latCard}{abs(hgtLat)}{lngCard}{abs(hgtLng)}.hgt'
            
            # Download data if path exists
            hgt_path = os.path.join(hgt_folder, hgt_file)
            if os.path.exists(hgt_path):
                
                # Read the binary elevation data from the SRTM HGT file
                with open(hgt_path, 'rb') as f:
                    data = f.read()
                    tile_data = np.frombuffer(data, dtype='>i2').reshape((NPOINTS, NPOINTS))
                    
                # Add to full matrix
                idxLat = all_elevations.shape[0] - ((NPOINTS-1)*(jLat+1)+1)
                idxLng = (NPOINTS-1)*jLng
                all_elevations[idxLat:idxLat+NPOINTS, idxLng:idxLng+NPOINTS] = tile_data
            else:
                raise FileNotFoundError(f'File {hgt_file} was not found.')
                
                    
    # Return elevations between GPS coords
    elevations = all_elevations[
        (NPOINTS-int((ne['lat']%1)*NPOINTS)):-int((sw['lat']%1)*NPOINTS),
        int((sw['lng']%1)*NPOINTS):-(NPOINTS-int((ne['lng']%1)*NPOINTS))
        ]
    return elevations


def estimate_map_size(sw, ne, MAX_SIZE=None, zoom=None):
    ''' For a given GPS region, either a zoom or MAX_SIZE must be specified. 
    If zoom is not specified, it is calculated such that the map is as large as
    possible without exceeding MAX_SIZE. The actual map dimensions can then
    be computed from the zoom.'''
    GLOBE_SIZE = 256 #size of globe google map at zoom 0

    # Google maps uses mercantor projection which changes spacing between 
    # lines of latitude. Extra math is needed to calculate latitude fraction.
    def latRad(lat):
        sin = np.sin(lat*np.pi/180)
        rad = np.log((1+sin)/(1-sin))/2
        return rad/2
    
    # Compute fractions relative to global range (180deg for latitude, 360deg for longitude)
    lat_fraction = (latRad(ne['lat']) - latRad(sw['lat']))/np.pi
    lng_fraction = (ne['lng'] - sw['lng'])/360

    if zoom is None:
        # Solve 2^(zoom) = SIZE / (frac*GLOBE_SIZE) for zoom
        lat_zoom = np.log(MAX_SIZE/GLOBE_SIZE/lat_fraction) / np.log(2)
        lng_zoom = np.log(MAX_SIZE/GLOBE_SIZE/lng_fraction) / np.log(2)
        zoom = int(min(lat_zoom, lng_zoom)) #round to integer minimal zoom

    # Solve 2^(zoom) = SIZE / (frac*GLOBE_SIZE) for SIZE
    map_height = int(2**zoom * GLOBE_SIZE * lat_fraction)
    map_width = int(2**zoom * GLOBE_SIZE * lng_fraction)

    return zoom, (map_height, map_width)


def download_tile(sw, ne, zoom, WATERMARK_PIXELS):
    API_KEY = os.environ['MAPS_API_KEY']
    
    # Determine size of map to download. Then add extra WATERMARK_PIXELS.
    _, (tile_height, tile_width) = estimate_map_size(sw, ne, zoom=zoom)
    size_string = f'{tile_width}x{tile_height+2*WATERMARK_PIXELS}'
    
    # Generate API call for a static map
    url = "https://maps.googleapis.com/maps/api/staticmap"
    
    lat_center = sw['lat']/2 + ne['lat']/2
    lng_center = sw['lng']/2 + ne['lng']/2
    center_string = f'{lat_center},{lng_center}'

    params = {
        "center": center_string,
        "zoom": zoom,
        "size": size_string,
        "maptype": "hybrid",
        "key": API_KEY,
        "style": {}
    }

    # Save map to file
    response = requests.get(url, params=params)
    with open('tile.png', 'wb') as f:
        f.write(response.content)
        
    return (tile_height, tile_width)


# %%
