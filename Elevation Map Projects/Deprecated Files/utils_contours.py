import cv2 as cv
import numpy as np
from scipy.interpolate import splprep, splev

def smoothContoursPolyDP(contours, error = .01):
    smoothed_contours = []
    for contour in contours:
        epsilon = error * cv.arcLength(contour, True)
        approx = cv.approxPolyDP(contour, epsilon, True)
        smoothed_contours.append(approx)
    return smoothed_contours

def smoothContoursInterpolate(contours, s=1.0, dist=10):
    smoothed = []
    for c in contours:
        x,y = c.T
        # Convert to python lists
        x = x.tolist()[0]
        y = y.tolist()[0] 

        tck, u = splprep([x,y], u=None, k=1, s=s, per=1)
        u_new = np.linspace(u.min(), u.max(), 4+int(cv.arcLength(c,True)/dist))
        x_new, y_new = splev(u_new, tck, der=0)
        res_array = [[[int(i[0]), int(i[1])]] for i in zip(x_new, y_new)]
        smoothed.append(np.asarray(res_array, dtype=np.int32))
    return smoothed

def morphKernel(pixels):
    return 255*np.ones((pixels, pixels), np.uint8)

def filterContours(contours, minArea = 50, maxArea=1e5):
    filtered_contours = []
    for c in contours:
        area = cv.contourArea(c)
        if area>minArea and area<maxArea:
            filtered_contours.append(c)

    return filtered_contours

def printContourSize(label, contours):
    totalSize = sum([len(c) for c in contours])
    print(f'Total Size for {label} = {totalSize}')
    return

def save2SVG(filename, contours, imgShape):
    h, w = imgShape

    with open(filename, "w+") as f:
        f.write(f'<svg width="{w}" height="{h}" xmlns="http://www.w3.org/2000/svg">')

        for c in contours:
            # Start drawing polygon
            f.write('<path d="M')
            for i in range(len(c)):
                x, y = c[i][0]
                f.write(f"{x} {y} ")
            # Complete polygon to initial point
            x, y = c[0][0]
            f.write(f"{x} {y} ")
            # End polygon
            f.write('" style="stroke:pink"/>')
        f.write("</svg>")

def showImage(img, max_size=1000):
    cv.imshow('test', resizeImg(img))
    cv.waitKey(0)
    cv.destroyAllWindows()   

def resizeImg(img, max_size=1000):
    resizedImg = img.copy()
    while resizedImg.shape[0]>max_size:
        resizedImg = cv.pyrDown(resizedImg)

    return resizedImg