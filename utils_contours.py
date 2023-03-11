import cv2 as cv
import numpy as np

def smoothContours(contours, error = .01):
    smoothed_contours = []
    for contour in contours:
        epsilon = error * cv.arcLength(contour, True)/len(contour)
        approx = cv.approxPolyDP(contour, epsilon, True)
        smoothed_contours.append(approx)
    return smoothed_contours

def morphKernel(pixels):
    return 255*np.ones((pixels, pixels), np.uint8)

def filterContours(contours, minArea = 50, maxArea=1e10):
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