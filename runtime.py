import os
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from collections import deque

# Hyperparameters

DISPLAY_RESOLUTION_SCALING = 0.5
DRAW_PATH_LENGTH = 100

lane_fov = [[590,550],[690,550], [1000,720],[200,720]]

hough_voting_minimum = 70

H_min = 0
H_max = 159
S_min = 0
S_max = 255
V_min = 190
V_max = 255

P_angle = 0.5
I_angle = 0.001
D_angle = 0.1

P_pos = 0.2
I_pos = 0.01
D_pos = 0.1

def show(mat):
    plt.imshow(cv.cvtColor(mat.astype(np.uint8), cv.COLOR_BGR2RGB))

def speedresize(frame):
    return cv.resize(frame, (int(frame.shape[1]*DISPLAY_RESOLUTION_SCALING), int(frame.shape[0]*DISPLAY_RESOLUTION_SCALING)))
    
def mask(imgshape):
    height = imgshape[0]
    width = imgshape[1]
    maskpts = np.array([[width//3, height//2], [2*width//3, height//2], [3*width//4, height], [width//4, height]])
    return cv.fillPoly(np.zeros(imgshape), pts=[maskpts], color=(255,255,255)).astype(np.uint8)

def correctionmask(imgshape, padding):
    padding = int(padding)
    height = imgshape[0]
    width = imgshape[1]
    maskpts = np.array([[width//3+padding, height//2+padding], [2*width//3-padding, height//2+padding], [3*width//4+padding, height], [width//4-padding, height]])
    return cv.fillPoly(np.zeros(imgshape), pts=[maskpts], color=(255,255,255)).astype(np.uint8)

def undistort(img, cMat, ncMat, dist, roi):
    dst = cv.undistort(img, cMat, dist, None, ncMat)
    x,y,w,h = roi
    dst = dst[y:y+h,x:x+w]
    return dst

def perspectiveTransform(image, pts, reverse=False):
    (tl, tr, br, bl) = pts

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # get new bounding box with respect to image size

    x_min = min(tl[0], bl[0])
    x_max = max(tr[0], br[0])
    y_min = min(tl[1], tr[1])
    y_max = max(bl[1], br[1])

    dst = np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]])

    if(reverse):
        M = cv.getPerspectiveTransform(dst, pts)
    else:
        M = cv.getPerspectiveTransform(pts, dst)

    warped = cv.warpPerspective(image, M, (image.shape[1], image.shape[0]))

    return warped

def houghlines(binary_image, draw_image, minimum_votes):
    """ Probablistic hough transform that returns estimated trajectory
    :param:
        Binary image
        image to draw lines on
        minimum hough space votes
        
    :return:
        average angular trajectory
        lowest x coordinate
    """
    lines = cv.HoughLinesP(binary_image, 1, np.pi / 180, minimum_votes, None, 50, 10)
    if lines is not None:
        angular_deviation = []
        lowest_point = []
        lowest_y = 0 # image y is flipped, so visually lowest is max value
        for i in range(len(lines)):
            l = lines[i][0]
            dx = l[0]-l[2]
            dy = l[1]-l[3]
            if l[1]>lowest_y:
                lowest_point = [l[0], l[1]]
                lowest_y = l[1]
            if l[3]>lowest_y:
                lowest_point = [l[2], l[3]]
                lowest_y = l[3]
            if dy==0:
                angular_deviation.append(np.pi/2)
            else:
                angular_deviation.append(np.arctan(dx/dy))
            cv.line(draw_image, (l[0], l[1]), (l[2], l[3]), (255,0,0), 3,  cv.LINE_AA)
        return sum(angular_deviation)/len(angular_deviation), lowest_point


## Camera calibration process

chessboardSize = (4,7)
calibrationImageSize=()

# get all calibration images

images = []
for path in os.listdir('calibration images'):
    if path.split('.')[1]=='jpg':
        img = cv.cvtColor(cv.imread('calibration images/'+path), cv.COLOR_BGR2GRAY)
        calibrationImageSize = img.shape
        images.append(img)

# find corners

objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2)

objpoints = []
imgpoints = []
for image in images:
    ret, corners = cv.findChessboardCorners(image, chessboardSize, None)
    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)

# calibration

ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, calibrationImageSize, None, None)

img = cv.imread('calibration images/calib1.jpg')
h,w = img.shape[:2]
newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(cameraMatrix, dist, (w,h), 1, (w,h))

capture = cv.VideoCapture('test.mp4')

CAPTURE_SHAPE = capture.read()[1].shape

self_pos = ()
self_angle = 0

int_pos = deque(maxlen=20)
int_angle = deque(maxlen=50)
prev_pos = -1
prev_angle = -1

angle = 0
marker = [0,0]

while True:
    ret, frame = capture.read()
    if not ret:
        break

    #frame = undistort(frame, cameraMatrix, newCameraMatrix, dist, roi)

    pts = np.array(lane_fov).astype(np.float32)

    warped_frame = perspectiveTransform(frame, pts)
    #warped_frame = undistort(warped_frame, cameraMatrix, newCameraMatrix, dist, roi)
    frame = cv.polylines(frame, [pts.reshape((-1,1,2)).astype(np.int32)], True, (255,0,255), 3)

    # get binary image, find contours and draw
    warped_binary = cv.inRange(cv.cvtColor(warped_frame, cv.COLOR_RGB2HSV), (H_min, S_min, V_min), (H_max, S_max, V_max))

    contours, h = cv.findContours(warped_binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    warped_contours = cv.drawContours(np.zeros(warped_binary.shape, dtype=np.uint8), contours, -1, (255,255,255), 3)

    # resizing and display
    frame = speedresize(frame)
    warped_frame = speedresize(warped_frame)
    warped_contours = speedresize(warped_contours)
    contoursbgr = cv.cvtColor(warped_contours, cv.COLOR_GRAY2BGR)
    
    
    # get hough lines and estimate target trajectory
    packet  = houghlines(warped_contours, warped_frame, hough_voting_minimum)
    if packet is not None:
        angle, marker = packet

    # draw target trajectory
    cv.drawMarker(warped_frame, marker, (0,255,0))
    cv.arrowedLine(warped_frame, marker, [int(marker[0]+DRAW_PATH_LENGTH*np.cos(angle+np.pi/2)), int(marker[1]-DRAW_PATH_LENGTH*np.sin(angle+np.pi/2))], (0,255,0), 2)

    
    # draw current trajectory

    if len(self_pos)==0:
        self_pos = [warped_frame.shape[1]//2, warped_frame.shape[0]]

    end_point = (int(self_pos[0]+DRAW_PATH_LENGTH*np.cos(self_angle+np.pi/2)), int(self_pos[1]-DRAW_PATH_LENGTH*np.sin(self_angle+np.pi/2)))
    warped_frame= cv.arrowedLine(warped_frame, self_pos, end_point, (0,0,255), 3)

    # calculate error

    angle_err = angle-self_angle
    pos_err = marker[0]-self_pos[0]

    # apply pid

    if prev_angle==-1:
        d_angle = 0
    else:
        d_angle = angle_err-prev_angle
    if prev_pos==-1:
        d_pos = 0
    else:
        d_pos = pos_err-prev_pos

    self_angle += (P_angle*angle_err + I_angle*sum(int_angle) + D_angle*d_angle)
    self_pos[0] += (P_pos*pos_err + I_pos*sum(int_pos) + D_pos*d_pos)
    self_pos[0] = int(self_pos[0])

    
    prev_pos = pos_err
    prev_angle = angle_err
    int_angle.append(angle_err)
    int_pos.append(pos_err)


    cv.imshow('frame', frame)
    cv.imshow('warped contours', warped_frame)
    if cv.waitKey(1) & 0xFF==ord('q'):
        break


