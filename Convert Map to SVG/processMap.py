
# %%
import cv2 as cv
import numpy as np
import utils_contours as uc


# Load an image
file = 'ski runs edited.png'
img = cv.imread(file)

footerRows = 80
minArea = 50
pixels = 10
pixels_to_thin_lines = 1


# Remove footer with Google watermarks
img = img[:-footerRows, :, :]


# Convert the image to grayscale, create white image
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
white = np.full_like(gray, 255, np.uint8)

# Find contours
contours, hierarchy = cv.findContours(gray, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
uc.printContourSize('gray', contours)

# Filter contours
contours = uc.filterContours(contours)
uc.printContourSize('filtered by area', contours)


contours = uc.smoothContoursInterpolate(contours, s=100, dist=100)
uc.printContourSize('Smoothed', contours)

# Interpolate
# contours = uc.smoothContours(contours)
# uc.printContourSize('smoothed contours', contours)



# Draw contours on a white image
cv.drawContours(white, contours, -1, (0, 0, 0), 2)


# Display the image
cv.imshow('Image', uc.resizeImg(white))
cv.imwrite('white.jpg', white)

# Save as svg
uc.save2SVG('test.svg', contours, white.shape)

# Wait for a key press and then close the window
cv.waitKey(0)
cv.destroyAllWindows()
# %%
