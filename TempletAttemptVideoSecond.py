import cv2
import pathlib
import time
import argparse
import os
import numpy as np

# Construct the argument parser.
parser = argparse.ArgumentParser()
parser.add_argument(
    '-i', '--input', help='path to input video',
    default='ImagesQuery/angleRightVideo.mp4'
)
parser.add_argument(
    '-t', '--template', help='path to the template',
    default='ImagesQuery/leftOnlyFullSize.png'
)
args = vars(parser.parse_args())

# Read the video input.
cap = cv2.VideoCapture(args['input'])
if (cap.isOpened() == False):
    print('Error while trying to read video. Please check path again')
# Get the frame width and height
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
# String name with which to save the resulting video.
save_name = str(pathlib.Path(
    args['input']
)).split(os.path.sep)[-1].split('.')[0]
# define codec and create VideoWriter object
out = cv2.VideoWriter(f"outputs/{save_name}.mp4",
                      cv2.VideoWriter_fourcc(*'mp4v'), 30,
                      (frame_width, frame_height))
# Read the template in grayscale format.
template = cv2.imread(args['template'], 0)
w, h = template.shape[::-1]
frame_count = 0 # To count total frames.
total_fps = 0 # To get the final frames per second.
# Read until end of video.
while (cap.isOpened()):
    # Capture each frame of the video.
    ret, frame = cap.read()
    if ret:
        image = frame.copy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        start_time = time.time()
        # Apply template Matching.
        res = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
        end_time = time.time()

        # Get the current fps.
        fps = 1 / (end_time - start_time)
        # Add `fps` to `total_fps`.
        total_fps += fps
        # Increment frame count.
        frame_count += 1

        threshold = 0.3

        # Store the coordinates of matched area in a numpy array
        loc = np.where(res >= threshold)

        # Draw a rectangle around the matched region.
        for pt in zip(*loc[::-1]):
            print("Found")
            cv2.rectangle(frame, pt, (pt[0] + w, pt[1] + h), (0, 255, 255), 2)




        cv2.imshow('Result', frame)
        out.write(frame)
        # Press `q` to exit.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
# Release VideoCapture() object.
cap.release()
# Close all frames and video windows.
cv2.destroyAllWindows()