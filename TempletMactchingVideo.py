from threading import Thread
import cv2
import numpy as np


thread1 = Thread()
thread1.start()
cap = cv2.VideoCapture(0)
searchFor = cv2.imread('ImagesQuery/angleLeftOnly.png', cv2.IMREAD_UNCHANGED)
# w, h = searchFor.shape[::-1]

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        thread1.join()
        break

    global grayFrame
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    graySearchFor = cv2.cvtColor(searchFor, cv2.COLOR_BGR2GRAY)

    # cv2.imshow('video gray', grayFrame)

    result = cv2.matchTemplate(grayFrame, graySearchFor, cv2.TM_CCOEFF_NORMED)

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    # print(max_loc)
    # print(max_val)
    cv2.imshow('video gray just', grayFrame)
    if max_val > 0.9:
        loc = np.where(result >= 0.9)
        print("Found it")
        for pt in zip(*loc[::-1]):
            print("Found")
            cv2.imshow('video gray', grayFrame)
            # cv2.rectangle(frame, pt, (pt[0] + w, pt[1] + h), (0, 255, 255), 2)
    elif max_val < 0.9:
        pass


cv2.waitKey(0)
cv2.destroyAllWindows()