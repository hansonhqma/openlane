import openlane as lib
import numpy as np
import cv2 as cv
import sys
from collections import deque
import time

# Telemetry

FRAMERATELOG = deque(maxlen=100)

# CV Hyperparameters

SOURCE = "testfootage.mov"
CALIBRATION_SOURCE = "calibration images"
CALIBRATION_BOARD_SIZE = (4,7)
INITIAL_FRAME = True

FRAME_SCALE = 1.5
TRANSFORM_SCALING = 0.5
TRASNFORM_VSHIFT = 30

BOX_COUNT = 4
BOX_WIDTH = 70
BOX_MAX = 3
LANE_BOXES = []
LANE_RESOLUTION = 5

# These two arrays are points on camera calibrated image!
TRANSFORM_PTS = np.array(([260,150],[395,150],[420,203],[244,203]))//FRAME_SCALE # corners of square on surface
MASK_PTS = np.array([[210, 90],[409,90],[619, 345],[0, 345]])//FRAME_SCALE # corners of mask
TRANSFORM_PTS = TRANSFORM_PTS.astype(np.int64)
MASK_PTS = MASK_PTS.astype(np.int64)


MARKER_COLOR = (255,255,0)
MARKER_SIZE = 2

HSV_MIN = (0,0,0)
HSV_MAX = (180,255,150)


ARGS = sys.argv
SHOWMASK = '-m' in ARGS
DRAWMARKERS = '-d' in ARGS

CALIBRATION_DATA = lib.getCameraMatrices(CALIBRATION_SOURCE, CALIBRATION_BOARD_SIZE)

# capture source

capture = cv.VideoCapture(SOURCE)

# Pre-process initial frame

ret, frame = capture.read()
if not ret: exit()

# lens undistortion -> preprocessing resize -> hsv threshold -> mask -> transform

frame = lib.undistortFrame(frame, *CALIBRATION_DATA)

frame = lib.fastresize(frame, 1/FRAME_SCALE)

binary_frame = cv.inRange(cv.cvtColor(frame, cv.COLOR_BGR2HSV), HSV_MIN, HSV_MAX)

binary_frame = lib.fovmask(binary_frame, MASK_PTS)

transform = lib.squarePerspectiveTransform(binary_frame, TRANSFORM_PTS, TRASNFORM_VSHIFT, SCALING=TRANSFORM_SCALING)

FRAME_HEIGHT = frame.shape[0]
FRAME_WIDTH = frame.shape[1]
BOX_HEIGHT = FRAME_HEIGHT//BOX_COUNT

# lane head search

lane_start_positions = lib.binaryImageHistogram(transform, BOX_COUNT, LANE_RESOLUTION)
for lane in lane_start_positions:
    LANE_BOXES.append([lane for x in range(BOX_COUNT)])

# lane finding stack
for lane in LANE_BOXES:
    lane[0] = lib.getBoundingBox(transform, lane[0], BOX_WIDTH, BOX_HEIGHT) # update bottom
    for i in range(1, len(lane)):
        if BOX_MAX!=-1 and i >= BOX_MAX:
            break
        bottom_center = [lane[i-1][0], lane[i-1][1]-BOX_HEIGHT] # calculate next box pos based on previous box
        lane[i] = lib.getBoundingBox(transform, bottom_center, BOX_WIDTH, BOX_HEIGHT)

while True:
    print(LANE_BOXES)
    loop_start_time = time.clock_gettime_ns(time.CLOCK_REALTIME)

    ret, frame = capture.read()
    if not ret: break

    # image processing stack
    frame = lib.undistortFrame(frame, *CALIBRATION_DATA)
    frame = lib.fastresize(frame, 1/FRAME_SCALE)

    binary_frame = cv.inRange(cv.cvtColor(frame, cv.COLOR_BGR2HSV), HSV_MIN, HSV_MAX)

    if SHOWMASK:
        frame = cv.polylines(frame, [MASK_PTS], True, (0,255,0))

    binary_frame = lib.fovmask(binary_frame, MASK_PTS)

    transform = lib.squarePerspectiveTransform(binary_frame, TRANSFORM_PTS, TRASNFORM_VSHIFT, SCALING=TRANSFORM_SCALING)

    if DRAWMARKERS:
        drawn_lane_markers = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3)).astype(np.uint8)
        raw_transform = lib.squarePerspectiveTransform(frame, TRANSFORM_PTS, TRASNFORM_VSHIFT, SCALING=TRANSFORM_SCALING)

    # lane finding stack
    for lane in LANE_BOXES:
        lane[0] = lib.getBoundingBox(transform, lane[0], BOX_WIDTH, BOX_HEIGHT) # update bottom
        if DRAWMARKERS:
            drawn_lane_markers = cv.circle(drawn_lane_markers, lane[0], MARKER_SIZE, MARKER_COLOR, -1)
            raw_transform = cv.rectangle(raw_transform, (lane[0][0]-BOX_WIDTH//2, lane[0][1]-BOX_HEIGHT), (lane[0][0]+BOX_WIDTH//2, lane[0][1]), (0,255,0))
        for i in range(1, len(lane)):
            if BOX_MAX!=-1 and i >= BOX_MAX:
                break
            bottom_center = [lane[i-1][0], lane[i-1][1]-BOX_HEIGHT] # calculate next box pos based on previous box
            lane[i] = lib.getBoundingBox(transform, bottom_center, BOX_WIDTH, BOX_HEIGHT)
            if DRAWMARKERS:
                drawn_lane_markers = cv.circle(drawn_lane_markers, lane[i], MARKER_SIZE, MARKER_COLOR, -1)
                raw_transform = cv.rectangle(raw_transform, (lane[i][0]-BOX_WIDTH//2, lane[i][1]-BOX_HEIGHT), (lane[i][0]+BOX_WIDTH//2, lane[i][1]), (0,255,0))

    if DRAWMARKERS:
        drawn_lane_markers = lib.squarePerspectiveTransform(drawn_lane_markers, TRANSFORM_PTS, TRASNFORM_VSHIFT, SCALING=TRANSFORM_SCALING, reverse=True)
        frame = cv.addWeighted(frame, 1, drawn_lane_markers, 1, 0)

    cv.imshow("Undistorted raw feed", frame)
    if DRAWMARKERS:
        cv.imshow("Raw transformed feed", raw_transform)

    cv.imshow("Binary transform", transform)

    time_delta = (time.clock_gettime_ns(time.CLOCK_REALTIME)-loop_start_time)/1000000000
    FRAMERATELOG.append(1/time_delta)
    
    if cv.waitKey(1) & 0xFF==ord('q'):
        break

print("Average fps: {:.2f}".format(sum(FRAMERATELOG)/len(FRAMERATELOG)))





