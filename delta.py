from pathlibcv import *
import cv2 as cv
import numpy as np


DISPLAY_SCALING = 0.6
capture = cv.VideoCapture("test.mp4")
first_frame = True

while(True):
    ret, frame = capture.read()
    if not ret:
        break

    # set dtype int64
    frame = cv.cvtColor(fastresize(frame, DISPLAY_SCALING), cv.COLOR_BGR2GRAY).astype(np.int64)

    if first_frame:
        # ensure previous frame exists
        first_frame = False
        prev_frame = frame
        continue
    
    delta_image = np.absolute(frame - prev_frame)
    
    prev_frame = frame
    
    delta_uint8 = delta_image.astype(np.uint8)
    
    cv.imshow("frame", delta_uint8)
    if cv.waitKey(1) & 0xFF==ord('q'):
        break
