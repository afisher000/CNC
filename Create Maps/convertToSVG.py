# %%
import cv2 as cv
import numpy as np
import utils_contours as uc
import utils_smoothing as us
from numpy.linalg import norm


# us.convert_map_to_SVG('missionridge_runs_edited.png', 'missionridge.svg', minArea=10)

# Load an image
file = 'missionridge_runs_edited.png'
gray = cv.imread(file, cv.IMREAD_GRAYSCALE)
white = np.full_like(gray, 255, np.uint8)

# Find contours, remove small/large areas
all_contours, hierarchy = cv.findContours(gray, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
areas = np.array([cv.contourArea(c) for c in all_contours])
minArea, maxArea = 40, 1e5
contours = np.delete(
    np.array(all_contours, dtype=object), 
    np.where((areas<minArea)|(areas>maxArea))[0],
    0
)
print(f'Reduced number of contours from {len(all_contours)} to {len(contours)}')
cv.drawContours(white, contours, -1, 0, -1)
cv.imwrite('contours.jpg', white)
# uc.showImage(white)

# Smooth each contour
smooth_contours = []

# for j, contour in enumerate([contours[31]]):
for j, contour in enumerate(contours):
    # Get points along line, remove duplicates near each other
    points = us.get_points_on_line(white, contour)
    points = us.remove_duplicate_points(points, min_dist=10)
    points = us.sort_line_points(points, max_sep=80, dist_power=5, theta_buffer=0.1, max_theta = 150)


    # Downsample curve and create new contour
    points = us.smooth_curve(points, 0.0001)
    smooth_contour = us.build_contour_from_points(points, linewidth=1)
    smooth_contours.append(smooth_contour)
    
    
    # print(j)
    # white = np.full_like(gray, 255, np.uint8)  
    # cv.drawContours(white, [contour], -1, 0, -1)
    # uc.showImage(white)
    # white = np.full_like(gray, 255, np.uint8)  
    # white = cv.drawContours(white, [smooth_contour], -1, 0, -1)
    # uc.showImage(white)

# Add border to svg
alignment = np.array([[[0,0]],[[0,1]],[[1,1]],[[1,0]]], dtype=np.int32)
size = 20
h, w = white.shape
smooth_contours.append(size*alignment)
smooth_contours.append(size*(alignment-1)+[w,h])


# Clean canvas and draw smooth contours
white = np.full_like(gray, 255, np.uint8)  
white = cv.drawContours(white, smooth_contours, -1, 0, -1)
uc.showImage(white)

print(f'Reduced total points from {sum(map(len, contours))} to {sum(map(len, smooth_contours))}')
uc.save2SVG('mission ridge.svg', smooth_contours, white.shape)






# %%
