# # Importing the libraries
# import cv2
#
# # Reading the image and converting into B/W
# image = cv2.imread('ImagesQuery/aLeftSmall.jpg')
# gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
# # Applying the function
# sift = cv2.xfeatures2d.SIFT_create()
# kp, des = sift.detectAndCompute(gray_image, None)
#
# # Applying the function
# kp_image = cv2.drawKeypoints(image, kp, None, color=(
#     0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# cv2.imshow('SIFT', kp_image)
# cv2.waitKey()

#-------------------------------------------------------------------------------------------------------------------

import cv2

#sift
sift = cv2.SIFT_create()

#feature matching
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)


img1 = cv2.imread('ImagesQuery/aRightSmall.jpg')
img2 = cv2.imread('ImagesQuery/angleLeftOnly.png')

img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

keypoints_1, descriptors_1 = sift.detectAndCompute(img1,None)
keypoints_2, descriptors_2 = sift.detectAndCompute(img2,None)

matches = bf.match(descriptors_1,descriptors_2)
matches = sorted(matches, key = lambda x:x.distance)


img3 = cv2.drawMatches(img1, keypoints_1, img2, keypoints_2, matches[:800], img2, flags=2)

cv2.imshow('SIFT', img3)

cv2.waitKey(0)