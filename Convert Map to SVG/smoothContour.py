# %%
import cv2 as cv
import numpy as np
import utils_contours as uc
import utils_smoothing as us
from numpy.linalg import norm

# Load an image
file = 'ski runs edited 2.png'
img = cv.imread(file)

# # Resize to correct aspect ratio ??
# sw = {'lat':47.273577, 'lng':-120.427392}
# ne = {'lat':47.291256, 'lng':-120.398520}
# current_ar = img.shape[0]/img.shape[1]
# ideal_ar = (ne['lat']-sw['lat'])/(ne['lng']-sw['lng'])

# # Erode image to make it easier to track line


#%%
# Remove footer with Google watermarks
footerRows = 80
img = img[:-footerRows, :, :]

# Read the image
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
white = np.full_like(gray, 255, np.uint8)

# Find contours, remove small/large areas
all_contours, hierarchy = cv.findContours(gray, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
areas = np.array([cv.contourArea(c) for c in all_contours])
minArea, maxArea = 200, 1e5
contours = np.delete(
    np.array(all_contours, dtype=object), 
    np.where((areas<minArea)|(areas>maxArea))[0],
    0
)
print(f'Reduced number of contours from {len(all_contours)} to {len(contours)}')
cv.drawContours(white, contours, -1, 0, -1)
# uc.showImage(white)

# Get smooth contours
smooth_contours = []
for contour in contours:

    # Get points along line, remove duplicates near each other
    points = us.get_points_on_line(white, contour)
    points = us.remove_duplicate_points(points, 7)
    
    # Build neighbor matrix and order points
    try:
        neighbors = us.get_neighbor_matrix(points, 400)
        points = us.order_centroids_by_neighbor(points, neighbors)
    except:
        us.draw_points(white, points, True)
        raise KeyError

    # Downsample curve and create new contour
    points = us.smooth_curve(points, 6)
    smooth_contour = us.build_contour_from_points(points)

    smooth_contours.append(smooth_contour)

# Clean canvas and draw smooth contours
white = np.full_like(gray, 255, np.uint8)  
white = cv.drawContours(white, smooth_contours, -1, 0, -1)
uc.showImage(white)

print(f'Reduced total points from {sum(map(len, contours))} to {sum(map(len, smooth_contours))}')
uc.save2SVG('mission ridge.svg', smooth_contours, white.shape)


# %%
