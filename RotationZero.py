# This programs calculates the orientation of an object.
# The input is an image, and the output is an annotated image
# with the angle of otientation for each object (0 to 180 degrees)

import cv2 as cv
from math import atan2, cos, sin, sqrt, pi
import numpy as np

# Load the image
img = cv.imread('ImagesQuery/aRightSmall.jpg')

# Was the image there?
if img is None:
    print("Error: File not found")
    exit(0)

cv.imshow('Input Image', img)

# Convert image to grayscale
# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
gray = cv.cvtColor(img, cv.COLOR_BGR2HSV)
lb = np.array([0, 0, 0])  # lower hsv bound for red
ub = np.array([186, 151, 255])  # upper hsv bound to red
mask = cv.inRange(gray, lb, ub)
res = cv.bitwise_and(img, img, mask=mask)

# Convert image to binary
_, bw = cv.threshold(mask, 50, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)

# Find all the contours in the thresholded image
contours, _ = cv.findContours(bw, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

for i, c in enumerate(contours):

    # Calculate the area of each contour
    area = cv.contourArea(c)

    # Ignore contours that are too small or too large
    if area < 3700 or 100000 < area:
        continue

    # cv.minAreaRect returns:
    # (center(x, y), (width, height), angle of rotation) = cv2.minAreaRect(c)
    rect = cv.minAreaRect(c)
    box = cv.boxPoints(rect)
    box = np.int0(box)

    # Retrieve the key parameters of the rotated bounding box
    center = (int(rect[0][0]), int(rect[0][1]))
    width = int(rect[1][0])
    height = int(rect[1][1])
    angle = int(rect[2])

    if width < height:
        angle = 90 - angle
    else:
        angle = -angle

    label = "  Rotation Angle: " + str(angle) + " degrees"
    textbox = cv.rectangle(img, (center[0] - 35, center[1] - 25),
                           (center[0] + 295, center[1] + 10), (255, 255, 255), -1)
    cv.putText(img, label, (center[0] - 50, center[1]),
               cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv.LINE_AA)
    cv.drawContours(img, [box], 0, (0, 0, 255), 2)

cv.imshow('Output Image', img)
cv.waitKey(0)
cv.destroyAllWindows()

# Save the output image to the current directory
cv.imwrite("min_area_rec_output.jpg", img)
