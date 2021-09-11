import pathlibcv as lib
import sys
from collections import deque
import time

# Telemetry

FRAMERATELOG = deque(maxlen=100)

# CV Hyperparameters

SOURCE = "path.mov"
CALIBRATION_SOURCE = "calibration images"
CALIBRATION_BOARD_SIZE = (4,7)
INITIAL_FRAME = True

BOX_COUNT = 6
BOX_WIDTH = 100
LANE_BOXES = []

FRAME_SCALE = 2
TRANSFORM_PTS = np.array(([267,139],[381,139],[399,192],[252,192])) # corners of square on surface\
TRANSFORM_SCALING = 0.8
MASK_PTS = np.array([[210, 90],[428,90],[640, 426],[0, 426]]) # corners of mask
MARKER_COLOR = (0,0,255)
MARKER_SIZE = 5

HSV_MIN = (0,0,0)
HSV_MAX = (180,255,150)

SHOWMASK = '-m' in sys.argv

CALIBRATION_DATA = lib.getCameraMatrices(CALIBRATION_SOURCE, CALIBRATION_BOARD_SIZE)

# capture source

capture = cv.VideoCapture(SOURCE)

# Pre-process initial frame

ret, frame = capture.read()
if not ret: exit()

frame = lib.fastresize(frame, 1/frame_scale)

FRAME_HEIGHT = frame.shape[0]
FRAME_WIDTH = frame.shape[1]
BOX_HEIGHT = FRAME_HEIGHT//box_count



while(True):
    loop_start_time = time.clock_gettime_ns(time.CLOCK_REALTIME)

    ret, frame = capture.read()
    if not ret: break

    frame = lib.fastresize(frame, 1/frame_scale)
