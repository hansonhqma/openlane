import cv2 as cv
import pathlibcv as lib
import os
import numpy as np
import time
from collections import deque

framerate = deque(maxlen=100)

cmat, ncmat, dist, roi = lib.getCameraMatrices('calibration images', (4,7))

capture = cv.VideoCapture(0)

frame, uframe = 0,0

while True:
    NS_TIME = time.clock_gettime_ns(time.CLOCK_REALTIME)
    ret, frame = capture.read()
    if not ret:
        break

    uframe = lib.undistort(frame, cmat, ncmat, dist, roi)

    #binary_image = lib.hsvThreshold(frame, (0,0,0), (180,255,150))

    cv.imshow('frame', frame)
    cv.imshow('uframe', uframe)

    TIME_DELTA = (time.clock_gettime_ns(time.CLOCK_REALTIME)-NS_TIME)/1000000000
    framerate.append(1/TIME_DELTA)

    if cv.waitKey(1) & 0xFF==ord('q'):
        break

print("Average fps: {:.2f}".format(sum(framerate)/len(framerate)))
print(frame.shape)
print(uframe.shape)
