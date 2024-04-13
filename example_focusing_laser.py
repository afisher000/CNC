# %%
from utils_gcode_2024 import gcode_manager
import numpy as np




nlines      = 21            # number of lines (best if odd so middle is z=0)
L           = 10            # line lengths
ysep        = 1             # line separation 
dz          = 20            # plus/minus z range to scan over

zs          = np.linspace(-dz, dz, nlines)
ys          = -1*np.arange(0, ysep*nlines, ysep)

# Create gcode
gm          = gcode_manager(feed_rate = 700)
for j in range(nlines):
    gm.add_path([0, L], [ys[j], ys[j]], [zs[j], zs[j]], laser=True)
gm.save('Examples/focusing_laser.nc')




# %%
