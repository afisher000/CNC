#%%
import numpy as np
import matplotlib.pyplot as plt

# In future, based on input latlong, open correct hgt_files for data.
# Allows multiple files. Tell user which files still need to be downloaded.
hgt_file = 'N47W121.hgt'
hgtLat = 47
hgtLon = -121 

npoints = 3601
LATS = np.linspace(hgtLat, hgtLat+1, 3601)
LONS = np.linspace(hgtLon,hgtLon+1, 3601)

# Read the binary data from the SRTM HGT file
with open(hgt_file, 'rb') as f:
    data = f.read()
ELEVATIONS = np.frombuffer(data, dtype='>i2').reshape((npoints, npoints))



def getElevation(lat, lon):
    lat_idx = npoints-int((lat-hgtLat)*npoints)
    lon_idx = int((lon-hgtLon)*npoints)
    return ELEVATIONS[lat_idx, lon_idx]
    
# Corner gps points of bounding rect
minLat, minLon = 47.27036, -120.43546
maxLat, maxLon = 47.295412, -120.394043

# Get lats and longs for image
imgLats = LATS[(LATS>minLat)&(LATS<maxLat)]
imgLons = LONS[(LONS>minLon)&(LONS<maxLon)]


img = np.zeros((len(imgLats), len(imgLons)))
elevationData = np.zeros_like(img)
for jlat, lat in enumerate(imgLats):
    for jlon, lon in enumerate(imgLons):
        elevationData[len(imgLats)-1-jlat, jlon] = getElevation(lat, lon)


# Scale elevations 
img = np.uint8(elevationData/elevationData.max()*255)
plt.imsave('elevationImage.png', img, cmap='gray')
print(f'Saved image with {len(imgLats), len(imgLons)} pixels')
# %%
