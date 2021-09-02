import os
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

def show(mat):
    plt.imshow(cv.cvtColor(mat.astype(np.uint8), cv.COLOR_BGR2RGB))
    
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

def perspectiveTransform(image, pts):
    (tl, tr, br, bl) = pts

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([[0, 0],[maxWidth - 1, 0],[maxWidth - 1, maxHeight - 1],[0, maxHeight - 1]], dtype = "float32")

    M = cv.getPerspectiveTransform(pts, dst)
    warped = cv.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped

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

capture = cv.VideoCapture('test2.mp4')

while True:
    ret, frame = capture.read()
    if not ret:
        break

    #frame = undistort(frame, cameraMatrix, newCameraMatrix, dist, roi)

    pts = np.array([[450,400],[730,400], [1130,700],[230,700]]).astype(np.float32)

    warped_frame = perspectiveTransform(frame, pts)
    frame = cv.polylines(frame, [pts.reshape((-1,1,2)).astype(np.int32)], True, (255,0,255))

    ret, warped_frame = cv.threshold(cv.cvtColor(warped_frame, cv.COLOR_BGR2GRAY), 100,255,0)

    contours, h = cv.findContours(warped_frame, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    warped_frame = cv.drawContours(np.zeros(warped_frame.shape, dtype=np.uint8), contours, -1, (255,255,255), 3)

    lines = cv.HoughLinesP(warped_frame, 1, np.pi / 180, 200, None, 50, 10)
    warped_frame = cv.cvtColor(warped_frame, cv.COLOR_GRAY2RGB)
    if lines is not None:
        for i in range(0, len(lines)):
            l = lines[i][0]
            cv.line(warped_frame, (l[0], l[1]), (l[2], l[3]), (0,0,255), 1,  cv.LINE_AA)


    cv.imshow('frame', frame)
    cv.imshow('warped frame', warped_frame)

    if cv.waitKey(1) & 0xFF==ord('q'):
        break
















