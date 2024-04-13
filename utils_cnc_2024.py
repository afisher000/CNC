import numpy as np

def get_raster_points(stepover, dimensions):
    Ny  = int(dimensions['height']/stepover)
    Nx  = int(dimensions['width']/stepover)

    unique_y = np.linspace(0, -dimensions['height'], Ny)
    unique_x = np.linspace(0, dimensions['width'], Nx)
    [xmat, ymat] = np.meshgrid(unique_x, unique_y)        
    
    # Flip X values in even rows so we cut in snake pattern
    xmat[1::2,:] = np.flip( xmat[1::2,:], axis=1)
    
    return xmat.ravel(), ymat.ravel()

def apply_interpolation(interp_fcn, xs, ys, dimensions):
    # Clip avoid model boundaries
    eps = 1e-6
    clipped_xs = np.clip(xs, eps, dimensions['width']-eps)
    clipped_ys = np.clip(ys, -dimensions['height']+eps, -eps)
    try:
        interp_fcn( (clipped_ys, clipped_xs) )
    except:
        print(clipped_xs)
    

    return interp_fcn( (clipped_ys, clipped_xs) )

def get_spiral_points(stepover, dimensions):
    Ny  = int(dimensions['height']/stepover)
    Nx  = int(dimensions['width']/stepover)

    unique_y = np.linspace(0, -dimensions['height'], Ny)
    unique_x = np.linspace(0, dimensions['width'], Nx)


    [xmat, ymat] = np.meshgrid(unique_x, unique_y)

    # Define pointers
    L, T = 0, 0
    B, R = len(unique_y)-1, len(unique_x)-1

    # Loop while more than one row or col left
    xs, ys = [], []
    while T<B and L<R:
        # Add four sides
        xs.extend(xmat[T:B, L])
        xs.extend(xmat[B, L:R])
        xs.extend(xmat[B:T:-1, R])
        xs.extend(xmat[T, R:L:-1])
        ys.extend(ymat[T:B, L])
        ys.extend(ymat[B, L:R])
        ys.extend(ymat[B:T:-1, R])
        ys.extend(ymat[T, R:L:-1])

        # increment pointers
        L += 1
        T += 1
        B -=1 
        R -=1

    if T==B: # single row left
        xs.extend(xmat[T, L:R+1])
        ys.extend(ymat[T, L:R+1])
    elif L==R: #single col left
        xs.extend(xmat[T:B+1, L])
        ys.extend(ymat[T:B+1, L])

    return np.array(xs), np.array(ys)

