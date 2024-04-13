
# %%
import os
import numpy as np
from pprint import pprint
import requests
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter
    
def compute_surface_interpolants(project, dimensions):
    # Get project coordinates and elevations
    sw, ne = get_GPS_coords(project)
    z_ft = get_elevation_data(sw, ne)

    # Apply transformation here
    if dimensions['rotate']==90:
        z_ft = np.rot90(z_ft)
    elif dimensions['rotate']==180:
        z_ft = np.rot90(np.rot90(z_ft))
    elif dimensions['rotate']==270:
        z_ft = np.rot90(z_ft, -1)
         
    # Define height of model and print
    if dimensions['rotate'] in [90, 270]:
        if hasattr(dimensions, 'width'):
            dimensions['height'] = dimensions['width'] * (ne['lng']-sw['lng']) / (ne['lat']-sw['lat'])
        else:
            dimensions['width'] = dimensions['height'] * (ne['lat']-sw['lat']) / (ne['lng']-sw['lng'])
    else:
        if hasattr(dimensions, 'width'):
            dimensions['height'] = dimensions['width'] * (ne['lat']-sw['lat']) / (ne['lng']-sw['lng']) 
        else:
            dimensions['width'] = dimensions['height'] * (ne['lng']-sw['lng']) / (ne['lat']-sw['lat'])
    
    print(f"\tDimensions:\n\t\tXwidth = {dimensions['width']:.1f}mm\n\t\tYheight = {dimensions['height']:.1f}mm\n\t\tZthickness = {dimensions['thickness']:.1f}mm")
    
    # Compute gradients    
    row_gradients = (z_ft[1:-1,2:] - z_ft[1:-1,:-2])/2
    col_gradients = (z_ft[2:,1:-1] - z_ft[:-2,1:-1])/2
    gradient_ft_by_px = np.pad(np.sqrt(row_gradients**2 + col_gradients**2), 1)
    
    # Conversions
    Ny, Nx          = z_ft.shape  # pixel counts in x and y
    elevft_to_mm    = dimensions['thickness']  / np.ptp(z_ft)
    Y_mm_to_px      = (Ny-1) / dimensions['height'] 
    X_mm_to_px      = (Nx-1) / dimensions['width']
    z_mm            = (z_ft-z_ft.max()) * elevft_to_mm

    # Get unitless gradient, smooth over couple datapoints to avoid noise
    gradients = gradient_ft_by_px * elevft_to_mm * X_mm_to_px 
    gradients = gaussian_filter(gradients, 2)

    # Define interpolations to map x,y [mm] to z [mm] or gradient [mm/mm]

    y_mm = -1*np.array(range(Ny)) * (1/Y_mm_to_px)
    x_mm = +1*np.array(range(Nx)) * (1/X_mm_to_px)

    # Have to use np.flip so grid vectors are increasing
    z_interp        = RegularGridInterpolator( (np.flip(y_mm), x_mm), np.flip(z_mm, axis=0) )
    gradient_interp = RegularGridInterpolator( (np.flip(y_mm), x_mm), np.flip(gradients, axis=0) )
    
    return z_interp, gradient_interp, dimensions

def get_GPS_coords(project):
    '''Reads GPS coords from file. Snaps to nearest latlng data points in data
    folders.'''
    NPOINTS = 3601
    
    # Read data, round to nearest arcsecond (1/3600 of degree)
    GPS_df = pd.read_csv('GPS coordinates.csv', index_col=0)
    GPS_coords = GPS_df.loc[project].apply(lambda x: (x//1) + int((x%1)*NPOINTS)/NPOINTS)
    
    # Write to sw and ne dictionaries
    sw, ne = {}, {}
    sw['lat'], sw['lng'], ne['lat'], ne['lng'] = GPS_coords
    return sw, ne


# def zinterpolation(xpoints, ypoints, MODEL, elevations):
#     # Define interpolation
#     elev_H, elev_W = elevations.shape
#     elev_interp = RegularGridInterpolator((range(elev_H), range(elev_W)), elevations)
    
#     # Convert points to elevation_indices
#     xindices        = xpoints / MODEL['W'] * (elev_W-1)
#     yindices        = -ypoints / MODEL['H'] * (elev_H-1)
#     elevation_values = elev_interp((yindices, xindices))

    
#     # Evaluate interpolation and convert to mm
#     zpoints = (elevation_values - elevations.max()) / np.ptp(elevations) * MODEL['T']
#     return zpoints


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
    API_KEY = os.environ['GOOGLE_API_KEY']
    
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

def download_satellite_map(project, dest_path):
    if os.path.exists(dest_path):
        print(f'\tFile "{dest_path}" already exists. Skipping\n')
        return
    
    sw, ne = get_GPS_coords(project)


    # Get satellite map from calls to Google Maps API
    MAX_MAP_SIZE = 2400                                                             # max size of full map
    WATERMARK_PIXELS = 60
    MAX_TILE_WIDTH = 640                                                            # max size of single google API call
    MAX_TILE_HEIGHT = MAX_TILE_WIDTH - 2*WATERMARK_PIXELS                           #Allow for cutting watermarked pixels
    MAX_TILES = 26



    # Determine latlngs for tiling
    zoom, (H, W) = estimate_map_size(sw, ne, MAX_MAP_SIZE)

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
            tile_H, tile_W = download_tile(tile_sw, tile_ne, zoom, WATERMARK_PIXELS)
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
    cv.imwrite(dest_path, resized_img)
