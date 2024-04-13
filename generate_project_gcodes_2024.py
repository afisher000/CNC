# %%
import utils_maps_2024 as umap
import utils_cnc_2024 as ucnc
import utils_svg_2024 as usvg
from utils_gcode_2024 import gcode_manager

import numpy as np
import os
import re

# Project and model size
project = 'altasnowbird'
dimensions = {
    'height':140,
    'thickness':30,
    'rotate':0,
}


# Create map if not exists
print(f'Downloading satelite image...')
dest_path = os.path.join('Projects', project, 'satellite.png')
umap.download_satellite_map(project, dest_path)

# Dimensions returned with height computed
print(f'Reading elevation data...')
z_interp, gradient_interp, dimensions = umap.compute_surface_interpolants(project, dimensions)



# %% Roughing
print(f'Generating roughing gcode...')
tool_diameter   = .25*25.4
stepover_frac   = 0.7         # Fraction of diameter that tool moves over each pass
max_stepdown    = 6            # Max change in z between passes
roughing_points = 5
stepover        = tool_diameter*stepover_frac

n_stepdowns     = int(np.ceil(dimensions['thickness']/max_stepdown))

# Get points of surface for roughing
xs, ys          = ucnc.get_spiral_points(stepover, dimensions)

# Z must be maximum of points within tool size
zs              = np.full_like(xs, fill_value=-dimensions['thickness'])
xy_offsets      = np.linspace(-tool_diameter/2, tool_diameter/2, roughing_points)
for dx in xy_offsets:
    for dy in xy_offsets:
        zsamples = ucnc.apply_interpolation(z_interp, xs+dx, ys+dy, dimensions)
        zs      = np.maximum(zs, zsamples)

# Create gcode
kwargs = {
    'feed_rate' : 2500,
    'zsafe'     : 5,
    'xsafe'     : -20
}
roughing        = gcode_manager(**kwargs)
for j in range(n_stepdowns):
    roughing.add_path(xs, ys, zs*(j+1)/n_stepdowns, roughing=True)
    roughing.lift() # finish raster at safe height

roughing.save( os.path.join('Projects', project, 'roughing.nc'))




# %% Smoothing
print(f'Generating smoothing gcode...')
tool_diameter   = 4
stepover_frac   = 0.3
stepover        = tool_diameter*stepover_frac

xs, ys          = ucnc.get_spiral_points(stepover, dimensions)
zs              = ucnc.apply_interpolation(z_interp, xs, ys, dimensions)
gradients       = ucnc.apply_interpolation(gradient_interp, xs, ys, dimensions)

# Correct zpoints with gradient information
zs              += (tool_diameter/2) * gradients / np.sqrt(1+gradients**2)

# Create gcode
smoothing       = gcode_manager(feed_rate=2500)
smoothing.add_path(xs, ys, zs)
smoothing.lift()
smoothing.save( os.path.join('Projects', project, 'smoothing.nc'))





# %% Runs Gcode
print(f'Generating Laser gcode')
svg_path        = os.path.join('Projects', project, 'runs.svg')
paths           = usvg.get_svg_paths(svg_path, dimensions)
xysep           = 0.25

# For each line, apply regex to look for X or Y value, then append z
laser           = gcode_manager(feed_rate = 700)

for path in paths:
    # if path[1]['stroke-width']>.3:
        # continue
    xs, ys      = usvg.add_points_to_path(path[0], max_distance = path[1]['stroke-width'])
    ys          = -ys  #convert inkscape coordinate system to cnc coordinate  system
    zs          = ucnc.apply_interpolation(z_interp, xs, ys, dimensions)

    # Assume thickness from stroke width
    thickness = np.ceil(path[1]['stroke-width']/.5)
    if thickness==1:
        laser.add_path(xs, ys, zs, laser=True, dasharray = path[1]['stroke-dasharray'])
    elif thickness==2:
        for du in [-xysep/2, xysep/2]:
            for dv in [-xysep/2, xysep/2]:
                dx = du - dv
                dy = du + dv
                laser.add_path(xs+dx, ys+dy, zs, laser=True, dasharray = path[1]['stroke-dasharray'])
    elif thickness==3:
        for du in [-xysep, 0, xysep]:
            for dv in [-xysep, 0, xysep]:
                dx = du - dv
                dy = du + dv
                laser.add_path(xs+dx, ys+dy, zs, laser=True, dasharray = path[1]['stroke-dasharray'])
laser.save(os.path.join('Projects', project, 'runs.nc'))





# %% Logo or signature
logo    = gcode_manager(feed_rate = 900)


# name    = 'logo'
# x0      = 14.45
# y0      = -(80.643+51.65)

name    = 'fisher'
x0      = 190
y0      = -(117.5+15.25)

# name    = 'text'
# x0      = 0
# y0      = -140.9


# Read gcode from lasergrbl file, add zs and offset
file    = os.path.join('Projects', project, f'{name}_lasergrbl.nc')
with open(file, 'r') as f:
    lines = f.readlines()

# Convert gcode to include elevation data
gcur, xcur, ycur = 0, 0, 0
for line in lines:
    if line[0]=='S':
        logo.gcode += line
    else:
        xmatch = re.search(r'X([-\d\.]+)', line)
        if xmatch:
            xcur = float(xmatch.group(1)) + x0

        ymatch = re.search(r'Y([-\d\.]+)', line)
        if ymatch:
            ycur = float(ymatch.group(1)) + y0

        gmatch = re.search(r'G(\d)', line)
        if gmatch:
            gcur = float(gmatch.group(1))

        zcur    = ucnc.apply_interpolation(z_interp, xcur, ycur, dimensions)
        logo.gcode += f'G{gcur:.0f} X{xcur:.2f} Y{ycur:.2f} Z{zcur:.2f}\n'
logo.save( os.path.join('Projects', project, f'{name}.nc'))







# %%
