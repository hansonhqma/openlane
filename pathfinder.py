import openlane as lib
from controller import controller
import numpy as np
import cv2 as cv
import sys
from collections import deque
import time



# Telemetry

FRAMERATELOG = deque(maxlen=100)

# Controller

pid = controller((250,100,300), (250,0,250), 0.3, 0.7)

VOLTAGE_SCALE = 1

# CV Hyperparameters

SOURCE = 0
CALIBRATION_SOURCE = "calibration images"
CALIBRATION_BOARD_SIZE = (4,7)
INITIAL_FRAME = True

FRAME_SCALE = 2
TRANSFORM_SCALING = 0.5
TRANSFORM_VSHIFT = 30

BOX_COUNT = 6
BOX_WIDTH = 60
BOX_MAX = 4
LANE_BOXES = []
LANE_RESOLUTION = 5

# These two arrays are points on camera calibrated image!
TRANSFORM_PTS = np.array(([225,150],[394,150],[404,203],[215,203]))//FRAME_SCALE # corners of square on surface
MASK_PTS = np.array([[0, 90],[619,90],[619, 345],[0, 345]])//FRAME_SCALE # corners of mask
TRANSFORM_PTS = TRANSFORM_PTS.astype(np.int64)
MASK_PTS = MASK_PTS.astype(np.int64)

MARKER_COLOR = (255,255,0)
MARKER_SIZE = 2

HSV_MIN = (0,0,0)
HSV_MAX = (180,255,120)

ARGS = sys.argv
DRIVE = '-drive' in ARGS
DEBUG = '-debug' in ARGS

if DRIVE:
    import motorcontrol

CALIBRATION_DATA = lib.getCameraMatrices(CALIBRATION_SOURCE, CALIBRATION_BOARD_SIZE)

# capture source

capture = cv.VideoCapture(SOURCE)

# wait out the exposure adjustment for ~50 frames

for i in range(50):
    ret, frame = capture.read()


# Pre-process initial frame

ret, frame = capture.read()
if not ret: exit()

# lens undistortion -> preprocessing resize -> hsv threshold -> mask -> transform

frame = lib.undistortFrame(frame, *CALIBRATION_DATA)

frame = lib.fastresize(frame, 1/FRAME_SCALE)

binary_frame = cv.inRange(cv.cvtColor(frame, cv.COLOR_BGR2HSV), HSV_MIN, HSV_MAX)

binary_frame = lib.fovmask(binary_frame, MASK_PTS)

transform = lib.squarePerspectiveTransform(binary_frame, TRANSFORM_PTS, TRANSFORM_VSHIFT, SCALING=TRANSFORM_SCALING)

FRAME_HEIGHT = frame.shape[0]
FRAME_WIDTH = frame.shape[1]
BOX_HEIGHT = FRAME_HEIGHT//BOX_COUNT


cc = ('M', 'J', 'P', 'G')
fps = 40
filetype = "avi"

raw = cv.VideoWriter("raw."+filetype, cv.VideoWriter_fourcc(*cc), fps, (FRAME_WIDTH, FRAME_HEIGHT))
rawtransform = cv.VideoWriter("rawtransform."+filetype, cv.VideoWriter_fourcc(*cc), fps, (FRAME_WIDTH, FRAME_HEIGHT))
binary = cv.VideoWriter("binary."+filetype, cv.VideoWriter_fourcc(*cc), fps, (FRAME_WIDTH, FRAME_HEIGHT))
vector = cv.VideoWriter("vector."+filetype, cv.VideoWriter_fourcc(*cc), fps, (FRAME_WIDTH, FRAME_HEIGHT))

# lane head search

lane_start_positions = lib.binaryImageHistogram(transform, BOX_COUNT, LANE_RESOLUTION)
for lane in lane_start_positions:
    LANE_BOXES.append([(lane[0], lane[1]-x*BOX_HEIGHT) for x in range(BOX_COUNT)])

lane = LANE_BOXES[0] # temporary solution

while True:
    loop_start_time = time.clock_gettime_ns(time.CLOCK_REALTIME)

    ret, frame = capture.read()
    if not ret: break

    # image processing stack
    isptime = time.clock_gettime_ns(time.CLOCK_REALTIME)
    frame = lib.undistortFrame(frame, *CALIBRATION_DATA)

    frame = lib.fastresize(frame, 1/FRAME_SCALE)

    binary_frame = cv.inRange(cv.cvtColor(frame, cv.COLOR_BGR2HSV), HSV_MIN, HSV_MAX)

    binary_frame = lib.fovmask(binary_frame, MASK_PTS)

    transform = lib.squarePerspectiveTransform(binary_frame, TRANSFORM_PTS, TRANSFORM_VSHIFT, SCALING=TRANSFORM_SCALING)

    if DEBUG:
        drawn_lane_markers = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3)).astype(np.uint8)
        raw_transform = lib.squarePerspectiveTransform(frame, TRANSFORM_PTS, TRANSFORM_VSHIFT, SCALING=TRANSFORM_SCALING)

    # lane finding stack
    lfstime = time.clock_gettime_ns(time.CLOCK_REALTIME)
    lane[0] = lib.getBoundingBox(transform, lane[0], BOX_WIDTH, BOX_HEIGHT) # update bottom
    if DEBUG:
        drawn_lane_markers = cv.circle(drawn_lane_markers, lane[0], MARKER_SIZE, MARKER_COLOR, -1)
        raw_transform = cv.rectangle(raw_transform, (lane[0][0]-BOX_WIDTH//2, lane[0][1]-BOX_HEIGHT), (lane[0][0]+BOX_WIDTH//2, lane[0][1]), (0,255,0))
    for i in range(1, len(lane)):
        if BOX_MAX!=-1 and i >= BOX_MAX:
            break
        bottom_center = [lane[i-1][0], lane[i-1][1]-BOX_HEIGHT] # calculate next box pos based on previous box
        lane[i] = lib.getBoundingBox(transform, bottom_center, BOX_WIDTH, BOX_HEIGHT)
        if DEBUG:
            drawn_lane_markers = cv.circle(drawn_lane_markers, lane[i], MARKER_SIZE, MARKER_COLOR, -1)
            raw_transform = cv.rectangle(raw_transform, (lane[i][0]-BOX_WIDTH//2, lane[i][1]-BOX_HEIGHT), (lane[i][0]+BOX_WIDTH//2, lane[i][1]), (0,255,0))

    vectorp1 = lane[0]
    vectorp2 = lane[BOX_MAX-1]
    arrow = cv.arrowedLine(np.zeros(frame.shape), vectorp1, vectorp2, (0,255,0), thickness=2)
    if vectorp2[0]-vectorp1[0]==0:
        angular_trajectory = 0
    else:
        angular_trajectory = np.arctan((vectorp2[0]-vectorp1[0])/(vectorp2[1]-vectorp1[1]))

    lateral_trajectory = 0.5 - vectorp1[0]/FRAME_WIDTH

    gain = pid.gain(angular_trajectory, lateral_trajectory, verbose=False)
    motorgain = pid.motorOutput(gain, VOLTAGE_SCALE)

    if DRIVE:
        motorcontrol.left.ChangeDutyCycle(motorgain[0])
        motorcontrol.right.ChangeDutyCycle(motorgain[1])

    if DEBUG:
        drawn_lane_markers = lib.squarePerspectiveTransform(drawn_lane_markers, TRANSFORM_PTS, TRANSFORM_VSHIFT, SCALING=TRANSFORM_SCALING, reverse=True)
        frame = cv.addWeighted(frame, 1, drawn_lane_markers, 1, 0)

        cv.imshow("Undistorted raw feed", frame)
        cv.imshow("Raw transformed feed", raw_transform)
        cv.imshow("trajectory", arrow)
        cv.imshow("Binary transform", transform)


    time_delta = (time.clock_gettime_ns(time.CLOCK_REALTIME)-loop_start_time)/1000000000
    FRAMERATELOG.append(1/time_delta)
    
    if cv.waitKey(1) & 0xFF==ord('q'):
        break

print("Average fps: {:.2f}".format(sum(FRAMERATELOG)/len(FRAMERATELOG)))
