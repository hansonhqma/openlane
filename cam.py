import cv2 as cv
import os
import numpy as np

def undistort(img, cMat, ncMat, dist, roi):
    dst = cv.undistort(img, cMat, dist, None, ncMat)
    x,y,w,h = roi
    dst = dst[y:y+h,x:x+w]
    return dst

capture = cv.VideoCapture(0)

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

while True:
    ret, frame = capture.read()
    if not ret:
        break

    frame = undistort(frame, cameraMatrix, newCameraMatrix, dist, roi)

    cv.imshow('frame', frame)
    if cv.waitKey(1) & 0xFF==ord('q'):
        break
