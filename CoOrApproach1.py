import numpy as np
import cv2
from tracker import *

# Load image, grayscale, Otsu's threshold, and extract ROI
image = cv2.imread(r'C:\Users\PC\Desktop\Dataset\unGate.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
x, y, w, h = cv2.boundingRect(thresh)
ROI = image[y:y+h, x:x+w]

# Color segmentation on ROI
hsv = cv2.cvtColor(ROI, cv2.COLOR_BGR2HSV)
lower = np.array([0, 0, 152])
upper = np.array([179, 255, 255])
mask = cv2.inRange(hsv, lower, upper)
res = cv2.bitwise_and(image, image, mask=mask)

# Crop left and right half of mask
x, y, w, h = 0, 0, ROI.shape[1]//2, ROI.shape[0]
left = mask[y:y+h, x:x+w]
right = mask[y:y+h, x+w:x+w+w]

#detection
obj_detector = cv2.createBackgroundSubtractorMOG2()
tracker = EuclideanDistTracker()
contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
detections = []
max_area = 0
maxAreaCo = [0, 0, 0, 0]
for cnt in contours:
    # Calculate area and remove small elements
    area = cv2.contourArea(cnt)
    if area > max_area:
        # cv2.drawContours(roi, [cnt], -1, (0, 255, 0), 2)
        x, y, w, h = cv2.boundingRect(cnt)
        detections.append([x, y, w, h])
        max_area = area
        maxAreaCo = [x, y, w, h]
boxes_ids = tracker.update(detections)

cv2.rectangle(res, (maxAreaCo[0], maxAreaCo[1]), (maxAreaCo[0] + maxAreaCo[2], maxAreaCo[1] + maxAreaCo[3]),
              (0, 255, 0), 3)







# Count pixels
left_pixels = cv2.countNonZero(left)
right_pixels = cv2.countNonZero(right)

print('Left pixels:', left_pixels)
print('Right pixels:', right_pixels)
#
# cv2.imshow('mask', mask)
cv2.imshow('mask', res)
# cv2.imshow('thresh', thresh)
# cv2.imshow('ROI', ROI)
# cv2.imshow('left', left)
# cv2.imshow('right', right)
cv2.waitKey()