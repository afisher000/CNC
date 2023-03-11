# %%
import cv2 as cv
import numpy as np
import utils_contours as uc


# Load an image
file = 'ski runs.png'
img = cv.imread(file)

footerRows = 80

# Remove footer with Google watermarks
img = img[:-footerRows, :, :]


# Convert the image to binary
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
ret, bin = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)
white = np.full_like(gray, 255, np.uint8)

# Fill small contours with white
contours, hierarchy = cv.findContours(gray, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
areas = np.array([cv.contourArea(c) for c in contours])
idxs = areas.argsort()

cv.drawContours(white, [contours[idxs[-4]]], -1, 0, 2)
# for c in contours:
#     area = cv.contourArea(c)
#     if area<500:
#         cv.drawContours(bin, [c], -1, 255, 5)

# uc.showImage(white)
cv.imwrite('test.jpg', white)


# # Apply Kmeans clustering
n_clusters = 100
ypoints, xpoints = np.where(white==0)
points = np.float32(np.vstack([ypoints, xpoints]).T)

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
_, _, rel_centers = cv.kmeans(points, n_clusters, None, criteria, 20, cv.KMEANS_RANDOM_CENTERS)
centers = rel_centers.astype(int)

# Draw centers (only those on a line)
for [y,x] in centers:
    if bin[y,x]==0:
        cv.circle(white, (x,y), 3, 155, 3)    
uc.showImage(white)
cv.imwrite('test.jpg', white)


e# %%
