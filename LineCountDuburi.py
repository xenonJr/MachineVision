import cv2
import numpy as np
count=True
count2=0
num2= False
a = []
count5=0
lu= False
lc= False
ld= False
fw=False
bw=False

cap = cv2.VideoCapture('ImagesQuery/GX010311.mp4')

# cap.set(3,640)
# cap.set(4,480)
v_width  = cap.get(3)  # float `width`
v_height = cap.get(4)  # float `height`

while True:
    ret, frame = cap.read()
    # frame2 = cv2.rotate(frame1, cv2.ROTATE_90_CLOCKWISE)
    # frame = cv2.rotate(frame2, cv2.ROTATE_90_CLOCKWISE)
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Red color
    low_red = np.array([100, 150, 0])
    high_red = np.array([140, 255, 255])
    red_mask = cv2.inRange(hsv_frame, low_red, high_red)
    red = cv2.bitwise_and(frame, frame, mask=red_mask)
    # Blue color
    # low_blue = np.array([100, 150, 0])
    # high_blue = np.array([140, 255, 255])
    low_blue = np.array([0, 0, 0])
    high_blue = np.array([255, 255, 121])
    # low_blue = np.array([0, 0, 0])
    # high_blue = np.array([76, 255, 121])

    blue_mask = cv2.inRange(hsv_frame, low_blue, high_blue)
    blue = cv2.bitwise_and(frame, frame, mask=blue_mask)
    contours, _ = cv2.findContours(blue_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    up_border = 0
    up_line=int((v_height/8)*1)
    mid_line =int(v_height/2)
    down_line=int((v_height/8)*7)
    down_border=v_height


    if len(contours) > 0:

        blue_area= max(contours, key=cv2.contourArea)
        (xg, yg, wg, hg) = cv2.boundingRect(blue_area)

    for contour in contours:

        area = cv2.contourArea(contour)
        #print(area)
        #print(xg,yg)

        if area > 80000:
            count=False
            #print(count)
            cv2.drawContours(frame, contour, -1, (0, 255, 0), 3)
            cv2.rectangle(frame, (xg,yg), (xg+wg,yg+hg), (0, 255, 0), 3)
            print(lu,lc,ld,fw,bw)
            my= (yg+yg+hg)/2

            # if (my < 120) and (my > 80):
            if (my < up_line) and (my > up_border):
                #print("first line")
                lu=True
                lc=False
                ld=False

                if (bw==True):
                    count5-=1
                    bw = False
                    ld = False
                    lc = False
            # if (my < 420) and (my> 380):
            if (my < mid_line+20) and (my > mid_line-20):
                #print("MID line")
                lc=True
                if  (lu==True):
                    fw=True
                    bw=False
                    lu=False
                if  (ld==True):
                    fw=False
                    bw=True
                    ld=False


            # if (my < 720) and (my > 680):
            if (my < down_border) and (my > down_line):
                #print("BOT line")
                ld=True
                lc = False
                lu = False

                if (fw==True) :
                    count5+=1
                    fw = False
                    lu = False
                    lc = False

            print(count5)
            x1 = wg / 2
            y1 = hg / 2
            cx = xg + x1
            cy = yg + y1
            a.append([cx, cy])
            #print(len(a))
            if (len(a) == 1):
                num2 = True

            a = []


            #print(len(a))


            #cv2.line(frame, (xg, wg), (xg + wg, yg+hg), (0, 255, 0), 3)
            cv2.line(frame, (0, up_line), (5000, up_line), (0, 0, 255), 3)
            cv2.line(frame, (0, mid_line ), (5000, mid_line), (0, 0, 255), 3)
            cv2.line(frame, (0, down_line), (5000, down_line), (0, 0, 255), 3)
            num = str(len(contours))

    # if (num2==True):
    #     count4+=1
    #     #print(count4)
    #     num2=False

            #print(count)
            # if(num==25):
            #     count=1
            #     print("count"+count)
            # elif(num==5):
            #     count=2
            #     print("count" + count)
            # elif(num==3):
            #     count=3
            #     print("count" + count)
            # else:
            #     count=0
            #     print("count" + count)




    # # This function allows us to create a descending sorted list of contour areas.
    # def contour_area(contours):
    #     cv2.drawContours(blue_mask, contours, -1, (0, 255, 255), 3)
    #
    #     # create an empty list
    #     cnt_area = []
    #
    #     # loop through all the contours
    #     for i in range(0, len(contours), 1):
    #         # for each contour, use OpenCV to calculate the area of the contour
    #         cnt_area.append(cv2.contourArea(contours[i]))
    #
    #     # Sort our list of contour areas in descending order
    #     list.sort(cnt_area, reverse=True)
    #     print(cnt_area)
    #     return cnt_area

    #
    #
    # def draw_bounding_box(contours, blue, number_of_boxes=1):
    #     # Call our function to get the list of contour areas
    #     cnt_area = contour_area(contours)
    #
    #     # Loop through each contour of our image
    #     for i in range(0, len(contours), 1):
    #         cnt = contours[i]
    #
    #         # Only draw the the largest number of boxes
    #         if (cv2.contourArea(cnt) > cnt_area[number_of_boxes]):
    #             # Use OpenCV boundingRect function to get the details of the contour
    #             x, y, w, h = cv2.boundingRect(cnt)
    #
    #             # Draw the bounding box
    #             image = cv2.rectangle(blue, (x, y), (x + w, y + h), (0, 0, 255), 2)
    #
    #     return image
    #
    #
    #     reddress = draw_bounding_box(contours, reddress)

    # Green color
    low_green = np.array([0, 0, 0])
    high_green = np.array([76, 255, 255])
    green_mask = cv2.inRange(hsv_frame, low_green, high_green)
    green = cv2.bitwise_and(frame, frame, mask=green_mask)

    # Every color except white
    low = np.array([0, 42, 0])
    high = np.array([100, 255, 255])
    mask = cv2.inRange(hsv_frame, low, high)
    result = cv2.bitwise_and(frame, frame, mask=mask)
    # cv2.imshow("Frame", frame)
    # cv2.imshow("Red", red)
    # cv2.imshow("Blue", blue)
    # cv2.imshow("Green", green)

    cv2.imshow("Result", cv2.resize(frame,(640,480)))


    key = cv2.waitKey(1)
    if key == 27:
        break