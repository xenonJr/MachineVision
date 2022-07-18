import cv2
import numpy as np
from tracker import *


def show_value(x):
    print("Lower Hue : " + str(cv2.getTrackbarPos("LH", "Tracking")))
    print("Lower Saturation : " + str(cv2.getTrackbarPos("LS", "Tracking")))
    print("Lower Value : " + str(cv2.getTrackbarPos("LV", "Tracking")))
    print("Upper Hue : " + str(cv2.getTrackbarPos("UH", "Tracking")))
    print("Upper Saturation : " + str(cv2.getTrackbarPos("US", "Tracking")))
    print("Upper Value : " + str(cv2.getTrackbarPos("UV", "Tracking")))


def nothing(x):
    pass


cap = cv2.VideoCapture(1);
obj_detector = cv2.createBackgroundSubtractorMOG2()
tracker = EuclideanDistTracker()

cv2.namedWindow("Tracking")
cv2.createTrackbar("LH", "Tracking", 0, 255, show_value)
cv2.createTrackbar("LS", "Tracking", 0, 255, show_value)
cv2.createTrackbar("LV", "Tracking", 0, 255, show_value)
cv2.createTrackbar("UH", "Tracking", 255, 255, show_value)
cv2.createTrackbar("US", "Tracking", 255, 255, show_value)
cv2.createTrackbar("UV", "Tracking", 255, 255, show_value)

while True:
    # frame = cv2.imread('smarties.png')
    _, frame = cap.read()

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    l_h = cv2.getTrackbarPos("LH", "Tracking")
    l_s = cv2.getTrackbarPos("LS", "Tracking")
    l_v = cv2.getTrackbarPos("LV", "Tracking")

    u_h = cv2.getTrackbarPos("UH", "Tracking")
    u_s = cv2.getTrackbarPos("US", "Tracking")
    u_v = cv2.getTrackbarPos("UV", "Tracking")

    l_b = np.array([l_h, l_s, l_v])
    u_b = np.array([u_h, u_s, u_v])

    lb = np.array([131, 32, 37])  # lower hsv bound for red
    ub = np.array([255, 255, 255])  # upper hsv bound to red

    lbG = np.array([30, 100, 100])  # lower hsv bound for red
    ubG = np.array([70, 255, 255])  # upper hsv bound to red

    mask = cv2.inRange(hsv, lb, ub)

    res = cv2.bitwise_and(frame, frame, mask=mask)

    # modified approach (worked)
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

    cv2.imshow("res", res)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
