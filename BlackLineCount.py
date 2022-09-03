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


def nothing():
    pass


def checkInitPos(x):
    if x > 150:
        return 1
    if x < 330:
        return 0



def printCount(x):
    cv2.putText(res, "Object Counter :" + str(x), (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)


# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('ImagesQuery/GX010311.mp4')
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
# cap.get(cv2.CV_CAP_PROP_FPS)
obj_detector = cv2.createBackgroundSubtractorMOG2()
tracker = EuclideanDistTracker()

c = 0
state = 0
thresh = 10
newArea = 0
initPos = 3
forwardCount = 0
backwardCount = 0
finalCount = 0
gotInitPos = False
willCount = False
countStatus = False
countStatus2 = False
position = 'Blank'

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


    frame2 = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    frame = cv2.rotate(frame2,cv2.ROTATE_90_CLOCKWISE)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    l_h = cv2.getTrackbarPos("LH", "Tracking")
    l_s = cv2.getTrackbarPos("LS", "Tracking")
    l_v = cv2.getTrackbarPos("LV", "Tracking")

    u_h = cv2.getTrackbarPos("UH", "Tracking")
    u_s = cv2.getTrackbarPos("US", "Tracking")
    u_v = cv2.getTrackbarPos("UV", "Tracking")

    l_b = np.array([l_h, l_s, l_v])
    u_b = np.array([u_h, u_s, u_v])

    lb = np.array([240, 100, 100])  # lower hsv bound for red
    ub = np.array([70, 255, 255])  # upper hsv bound to red
    lower_blue = np.array([100, 150, 0])
    upper_blue = np.array([140, 255, 255])

    lbG = np.array([30, 100, 100])  # lower hsv bound for red
    ubG = np.array([70, 255, 255])  # upper hsv bound to red
    # hsv(216, 93, 75)

    lbBlack = np.array([0, 0, 0])
    ubBlack = np.array([255, 236, 125])

    mask = cv2.inRange(hsv, lbBlack, ubBlack)

    res = cv2.bitwise_and(frame, frame, mask=mask)

    willCount = False

    # modified approach (worked)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    max_area = 0
    maxAreaCo = [0, 0, 0, 0]
    for cnt in contours:
        # Calculate area and remove small elements
        area = cv2.contourArea(cnt)
        if area == 0:
            countStatus = countStatus2
            countStatus2 = False
            willCount = False

            # state -= 1
            # if state < -thresh:
            #     state = thresh
            # if state < -30:
            #     countStatus = countStatus2
            #     countStatus2 = False

        if area > 80000:
            gotInitPos = True
            # cv2.drawContours(roi, [cnt], -1, (0, 255, 0), 2)
            x, y, w, h = cv2.boundingRect(cnt)
            detections.append([x, y, w, h])
            max_area = area
            maxAreaCo = [x, y, w, h]
            countStatus = countStatus2
            countStatus2 = True
            willCount = True
            # state += 1
            # if state > thresh:
            #    state = thresh
            # if state > 30:
            #     countStatus = countStatus2
            #     countStatus2 = True


    detectedWidth = maxAreaCo[0] + maxAreaCo[2]
    detectedHeight = maxAreaCo[1] + maxAreaCo[3]

    start_point = (0, 540)
    end_point = (1920, 540)
    color = (0, 255, 0)
    thickness = 9

    cv2.line(res, start_point, end_point, color, thickness)


    if initPos == 3:
        if gotInitPos == True:
            initPos = checkInitPos(detectedHeight)
            gotInitPos = False


    if detectedHeight > 540:
        if detectedHeight < 555:
            if willCount == True:
                forwardCount = forwardCount + 1


    # if detectedHeight < 240:
    #     if detectedHeight > 225:
    #         if willCount == True:
    #             backwardCount = backwardCount + 1

    if willCount == False:
        if forwardCount > 0:
            if initPos == 0:
                finalCount = finalCount - 1
                forwardCount = 0
            if initPos == 1:
                finalCount = finalCount + 1
                forwardCount = 0


        # if backwardCount > 0:
        #     finalCount = finalCount-1
        #     backwardCount = 0

    if detectedHeight < 240:
                position = 'Top'
                cv2.putText(res, "TOP", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
                cv2.putText(res, "Height :" + str(detectedHeight), (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
                cv2.putText(res, "Width :" + str(detectedWidth), (50, 170), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
    elif detectedHeight > 240:
                position = 'Bottom'
                cv2.putText(res, "BOTTOM", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
                cv2.putText(res, "Height :" + str(detectedHeight), (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
                cv2.putText(res, "Width :" + str(detectedWidth), (50, 170), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)

    else:
            cv2.putText(res, "NOT FOUND", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)


    if countStatus == True:
        if countStatus2 == False:
            if position == 'Top':
                c = c-1
            if position == 'Bottom':
                c = c+1
            position = 'BLANK'

    boxes_ids = tracker.update(detections)

    cv2.rectangle(res, (maxAreaCo[0], maxAreaCo[1]), (maxAreaCo[0] + maxAreaCo[2], maxAreaCo[1] + maxAreaCo[3]),
                  (0, 255, 0), 3)

    cv2.putText(res, "Object Count : " + str(finalCount), (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
    # cv2.putText(res, "State Count : " + str(state), (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
    cv2.putText(res, "Init Pos : " + str(initPos), (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)

    cv2.imshow("res", res)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
