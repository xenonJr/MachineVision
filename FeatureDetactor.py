import os

import numpy as np
import cv2
from tracker import *

obj_detector = cv2.createBackgroundSubtractorMOG2()
tracker = EuclideanDistTracker()

img1 = cv2.imread('ImagesQuery/robosub.png', 0)
img2 = cv2.imread('ImagesTrain/robosubTrain.PNG', 0)

orb = cv2.ORB_create(nfeatures=5000)

kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

matches = cv2.BFMatcher().knnMatch(des1, des2, k=2)

good = [[m] for m, n in matches if m.distance < 0.9 * n.distance]
print(len(good))
img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=0)



# img4 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG)

# cv2.drawKeypoints(img2, kp2, outImage=img2, color=(255, 0, 0),
#                   flags=cv2.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG)

#
# contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# detections = []
# max_area = 0
# maxAreaCo = [0, 0, 0, 0]
# for cnt in contours:
#     # Calculate area and remove small elements
#     area = cv2.contourArea(cnt)
#     if area > max_area:
#         # cv2.drawContours(roi, [cnt], -1, (0, 255, 0), 2)
#         x, y, w, h = cv2.boundingRect(cnt)
#
#         detections.append([x, y, w, h])
#         max_area = area
#         maxAreaCo = [x, y, w, h]
#     boxes_ids = tracker.update(detections)
#
# cv2.rectangle(img3, (maxAreaCo[0], maxAreaCo[1]), (maxAreaCo[0] + maxAreaCo[2], maxAreaCo[1] + maxAreaCo[3]),
#               (0, 255, 0), 3)

# imgKp1 = cv2.drawKeypoints(img1, kp1, None)
# imgKp2 = cv2.drawKeypoints(img2, kp2, None)

# cv2.imshow('Kp1', imgKp1)
# cv2.imshow('Kp2', imgKp2)
cv2.imshow('img1', img1)
cv2.imshow('img2', img2)
cv2.imshow('img3', img3)
# cv2.imshow('img4', img4)

cv2.waitKey(0)
