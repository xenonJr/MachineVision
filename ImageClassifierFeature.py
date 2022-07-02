import os

import numpy as np
import cv2

path = 'ImagesQuery'
orb = cv2.ORB_create(nfeatures=5000)

images = []
classNames = []
mylist = os.listdir(path)
print(mylist)
print('Total Classes Detected ', len(mylist))

for cl in mylist:
    imgCur = cv2.imread(f'{path}/{cl}', 0)
    images.append(imgCur)
    classNames.append(os.path.splitext(cl[0]))
print(classNames)


def findDes(images):
    desList = []
    for img in images:
        kp, des = orb.detectAndCompute(img, None)
        desList.append(des)

    return desList


def findID(img, desList, thress=700):
    kp2, des2 = orb.detectAndCompute(img, None)
    matchList = []
    try:
        for des in desList:
            matches = cv2.BFMatcher().knnMatch(des, des2, k=2)
            good = [[m] for m, n in matches if m.distance < 0.9 * n.distance]
            matchList.append(len(good))
        print(matchList)
    except:
        pass
    finalVal = -1
    if len(matchList) != 0:
        if max(matchList) > thress:
            finalVal = matchList.index(max(matchList))

    return finalVal


desList = findDes(images)
print(len(desList))

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

while True:
    success, img2 = cap.read()
    imgOrginal = img2.copy()
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    id = findID(img2, desList)
    if id != -1:
        cv2.putText(imgOrginal, "Found", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)

    cv2.imshow('img3', imgOrginal)
    cv2.waitKey(1)

# matches = cv2.BFMatcher().knnMatch(des1, des2, k=2)
#
# good = [[m] for m, n in matches if m.distance < 0.9 * n.distance]
# print(len(good))
# img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=0)
#
# # imgKp1 = cv2.drawKeypoints(img1, kp1, None)
# # imgKp2 = cv2.drawKeypoints(img2, kp2, None)
#
# # cv2.imshow('Kp1', imgKp1)
# # cv2.imshow('Kp2', imgKp2)
# cv2.imshow('img1', img1)
# cv2.imshow('img2', img2)
# cv2.imshow('img3', img3)

cv2.waitKey(0)
cv2.destroyAllWindows()
