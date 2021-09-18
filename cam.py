import cv2 as cv
import openlane as lib
import os
import sys
import numpy as np
import time
from collections import deque

framerate = deque(maxlen=100)

cmat, ncmat, dist, roi = lib.getCameraMatrices('calibration images', (4,7))

capture = cv.VideoCapture(int(sys.argv[1]))

while True:
    NS_TIME = time.clock_gettime_ns(time.CLOCK_REALTIME)
    ret, frame = capture.read()
    if not ret:
        break

    uframe = lib.undistortFrame(frame, cmat, ncmat, dist, roi)

    cv.imshow('uframe', uframe)

    TIME_DELTA = (time.clock_gettime_ns(time.CLOCK_REALTIME)-NS_TIME)/1000000000
    framerate.append(1/TIME_DELTA)

    if cv.waitKey(1) & 0xFF==ord('q'):
        break

print("Average fps: {:.2f}".format(sum(framerate)/len(framerate)))
print(uframe.shape)
