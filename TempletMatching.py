# Python program to illustrate
# template matching
import cv2
import numpy as np

# Read the main image
img_rgb = cv2.imread('ImagesQuery/aRightSmall.jpg')

# Convert it to grayscale
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

# Read the template
template = cv2.imread('ImagesQuery/rightOnlySmall.png', 0)

# Store width and height of template in w and h
w, h = template.shape[::-1]

# Perform match operations.
res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)

# Specify a threshold
# threshold = 0.39 works with only left.
threshold = 0.1

# Store the coordinates of matched area in a numpy array
loc = np.where(res >= threshold)

# Draw a rectangle around the matched region.
for pt in zip(*loc[::-1]):
	print("Found")
	cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,255,255), 2)

# Show the final image with the matched area.
cv2.imshow('Detected', img_rgb)
# imS = cv2.resize(img_rgb, (960, 540))                # Resize image
# cv2.imshow("output", imS)
cv2.waitKey(0)
cv2.destroyAllWindows()
