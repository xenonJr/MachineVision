import cv2

videoCaptureObject = cv2.VideoCapture(1)


def sketch_transform(image):  # in here you can do image filteration
    image_grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_grayscale_blurred = cv2.GaussianBlur(image_grayscale, (7, 7), 0)
    image_canny = cv2.Canny(image_grayscale_blurred, 10, 80)
    _, mask = image_canny_inverted = cv2.threshold(image_canny, 30, 255, cv2.THRESH_BINARY_INV)
    return mask


upper_left = (0, 0)
bottom_right = (320, 480)

result = True
while (result):
    ret, image_frame = videoCaptureObject.read()

    # Rectangle marker
    r = cv2.rectangle(image_frame, upper_left, bottom_right, (100, 50, 200), 5)
    rect_img = image_frame[upper_left[1]: bottom_right[1], upper_left[0]: bottom_right[0]]

    sketcher_rect = rect_img
    sketcher_rect = sketch_transform(sketcher_rect)

    # Conversion for 3 channels to put back on original image (streaming)
    sketcher_rect_rgb = cv2.cvtColor(sketcher_rect, cv2.COLOR_GRAY2RGB)

    # Replacing the sketched image on Region of Interest
    image_frame[upper_left[1]: bottom_right[1], upper_left[0]: bottom_right[0]] = sketcher_rect_rgb

    cv2.imshow("test", image_frame)
    k = cv2.waitKey(1)

    if k % 256 == 27:
        print("Escape hit, closing...")
        break
    elif k % 256 == 32:
        cv2.imwrite("input_image.jpg", image_frame)
        img = cv2.imread("input_image.jpg")

        # you can do those function inside the sketch_transform def

        # gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # gray_img = cv2.resize(gray_image, (28, 28)).reshape(1,28,28,1)
        # ax[2,1].imshow(gray_img.reshape(28, 28) , cmap = "gray")
        # cv2.imshow("image", gray_img.reshape(28, 28))

        y_pred = model.predict_classes(img)
        print("predicted alphabet  = ", y_pred)
        # text_to_audio(myDict.get(y_pred[0]))
videoCaptureObject.release()
cv2.destroyAllWindows()