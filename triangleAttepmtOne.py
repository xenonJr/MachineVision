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


cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture(r'C:\Users\PC\Desktop\ballpRe\mvd\2.mp4')
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

obj_detector = cv2.createBackgroundSubtractorMOG2()
tracker = EuclideanDistTracker()

if cap.isOpened(): # try to get the first frame
    rval, frame = cap.read()
else:
    rval = False
cv2.namedWindow("Tracking")
cv2.createTrackbar("LH", "Tracking", 0, 255, show_value)
cv2.createTrackbar("LS", "Tracking", 0, 255, show_value)
cv2.createTrackbar("LV", "Tracking", 0, 255, show_value)
cv2.createTrackbar("UH", "Tracking", 255, 255, show_value)
cv2.createTrackbar("US", "Tracking", 255, 255, show_value)
cv2.createTrackbar("UV", "Tracking", 255, 255, show_value)

#rectangle
upper_left = (34, 480)
bottom_right = (310, 210)
p_img = np.zeros((640, 480), dtype=np.uint8)
h_img = np.zeros((640, 480), dtype=np.uint8)


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

    lbBall = np.array([9, 139, 95])  # lower hsv bound for red
    ubBall = np.array([255, 255, 255])  # upper hsv bound to red

    mask = cv2.inRange(hsv, lb, ub)
    # mask = cv2.inRange(hsv, lb, ub)

    res = cv2.bitwise_and(frame, frame, mask=mask)

    # r = cv2.rectangle(res, upper_left, bottom_right, (100, 50, 200), 5)
    rect_img = res[upper_left[1]: bottom_right[1], upper_left[0]: bottom_right[0]]

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
    detectedTopLeft = (maxAreaCo[0], maxAreaCo[1])
    detectedWidth = maxAreaCo[0] + maxAreaCo[2]
    detectedHeight = maxAreaCo[1] + maxAreaCo[3]
    # if detectedWidth > 310 & detectedHeight < 240:
    #     cv2.putText(res, "TOP RIGHT", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
    #     cv2.putText(res, "Height :" + str(detectedHeight), (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
    #     cv2.putText(res, "Width :" + str(detectedWidth), (50, 170), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
    # elif detectedWidth < 310 & detectedHeight > 240:
    #     cv2.putText(res, "BOTTOM LEFT", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
    #     cv2.putText(res, "Height :" + str(detectedHeight), (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
    #     cv2.putText(res, "Width :" + str(detectedWidth), (50, 170), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
    # elif detectedWidth < 300 & detectedHeight < 240:
    #     cv2.putText(res, "TOP LEFT", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
    #     cv2.putText(res, "Height :" + str(detectedHeight), (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
    #     cv2.putText(res, "Width :" + str(detectedWidth), (50, 170), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
    # elif detectedWidth > 310 & detectedHeight > 240:
    #     cv2.putText(res, "BOTTOM RIGHT", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
    #     cv2.putText(res, "Height :" + str(detectedHeight), (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
    #     cv2.putText(res, "Width :" + str(detectedWidth), (50, 170), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)

    if detectedWidth < 310:
            if detectedHeight < 240:
                cv2.putText(res, "TOP LEFT", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
                cv2.putText(res, "Height :" + str(detectedHeight), (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
                cv2.putText(res, "Width :" + str(detectedWidth), (50, 170), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
            elif detectedHeight > 240:
                cv2.putText(res, "BOTTOM LEFT", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
                cv2.putText(res, "Height :" + str(detectedHeight), (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
                cv2.putText(res, "Width :" + str(detectedWidth), (50, 170), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
    elif detectedWidth > 310:
            if   detectedHeight < 240:
                cv2.putText(res, "TOP RIGHT", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
                cv2.putText(res, "Height :" + str(detectedHeight), (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
                cv2.putText(res, "Width :" + str(detectedWidth), (50, 170), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
            elif  detectedHeight > 240:
                cv2.putText(res, "BOTTOM RIGHT", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
                cv2.putText(res, "Height :" + str(detectedHeight), (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
                cv2.putText(res, "Width :" + str(detectedWidth), (50, 170), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)

    else:
            cv2.putText(res, "NOT FOUND", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)

    detectedWidthHeight = (maxAreaCo[0] + maxAreaCo[2], maxAreaCo[1] + maxAreaCo[3])
    cv2.rectangle(res, detectedTopLeft, detectedWidthHeight,
                  (0, 255, 0), 3)

    cv2.imshow("res", res)


    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
