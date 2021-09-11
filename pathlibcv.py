import cv2 as cv
import numpy as np
import os

def show(img, caption="frame"):
    cv.imshow(caption, img)
    cv.waitKey(0)
    cv.destroyAllWindows()

def drawMask(image, pts, outline_source = False):
    mask = cv.fillPoly(np.zeros(image.shape).astype(np.uint8), pts=[pts], color=(255,255,255)).astype(np.uint8)
    if outline_source:
        image = cv.polylines(image, [pts.reshape((-1,1,2))], True, (255,255,0), 2)
    return cv.bitwise_and(image, mask)

def quickMask(imgshape):
    height = imgshape[0]
    width = imgshape[1]
    maskpts = np.array([[width//3, height//2], [2*width//3, height//2], [3*width//4, height], [width//4, height]])
    return cv.fillPoly(np.zeros(imgshape), pts=[maskpts], color=(255,255,255)).astype(np.uint8)

def quickCorrectionmask(imgshape, padding):
    padding = int(padding)
    height = imgshape[0]
    width = imgshape[1]
    maskpts = np.array([[width//3+padding, height//2+padding], [2*width//3-padding, height//2+padding], [3*width//4+padding, height], [width//4-padding, height]])
    return cv.fillPoly(np.zeros(imgshape), pts=[maskpts], color=(255,255,255)).astype(np.uint8)

def fastresize(frame, DISPLAY_RESOLUTION_SCALING):
    """ Simple resize function
    args:
        frame: image to be resized
        DISPLAY_RESOLUTION_SCALING: non-negative scaling factor

    returns:
        resized image
    """
    return cv.resize(frame, (int(frame.shape[1]*DISPLAY_RESOLUTION_SCALING), int(frame.shape[0]*DISPLAY_RESOLUTION_SCALING)))

def hsvThreshold(img, minvalues, maxvalues):
    hsvcolor = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    return cv.inRange(hsvcolor, minvalues, maxvalues)


def undistort(img, cMat, ncMat, dist, roi):
    """ Camera undistortion function that applies an undistortion matrix
    args:
        img: image to be undistorted

    returns:
        undistorted image

    """
    dst = cv.undistort(img, cMat, dist, None, ncMat)
    x,y,w,h = roi
    dst = dst[y:y+h,x:x+w]
    return dst

def getCameraMatrices(folderpath, chessboardSize):
    """ Returns original and optimized camera matrices, dist, and roi used for image undistortion
    args:
        folderpath: filepath to folder containing calibration images
        chessboardSize: dimensions of chess board by internal corners, cols x rows
    """
    calibrationImageSize = ()
    images = []
    for path in os.listdir(folderpath):
        if path.split('.')[1]=='jpg':
            img = cv.cvtColor(cv.imread(folderpath+'/'+path), cv.COLOR_BGR2GRAY)
            calibrationImageSize = img.shape
            images.append(img)
    if len(images)==0:
        print("No calibration images found!")
        exit(-1)

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

    h,w = images[0].shape[:2]
    newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(cameraMatrix, dist, (w,h), 1, (w,h))

    return cameraMatrix, newCameraMatrix, dist, roi


def squarePerspectiveTransform(image, pts, SCALING=1, reverse=False):
    """ Performs a 4 point perspective warp on an image, assuming pts marks out the corners of a flat square
    args:
        image: source image
        pts: polygon of region to be mapped to rectangle
        reverse: True if the perspective transform is to be reverse, false else and default
    
    returns:
        transformed image

    """
    pts = pts.astype(np.float32)
    (tl, tr, br, bl) = pts

    # get new bounding box with respect to marked area

    x_min = min(tl[0], bl[0])
    x_max = max(tr[0], br[0])
    y_min = min(tl[1], tr[1])
    y_max = max(bl[1], br[1])

    # calculate square bounding box

    square_edge_distance = (x_max-x_min)*SCALING

    bounding_box_center = ((x_max+x_min)//2, (y_max+y_min)//2+60)

    dst = np.array([[bounding_box_center[0]-square_edge_distance//2, bounding_box_center[1]-square_edge_distance//2],
        [bounding_box_center[0]+square_edge_distance//2, bounding_box_center[1]-square_edge_distance//2],
        [bounding_box_center[0]+square_edge_distance//2, bounding_box_center[1]+square_edge_distance//2],
        [bounding_box_center[0]-square_edge_distance//2, bounding_box_center[1]+square_edge_distance//2]]).astype(np.float32)

    if(reverse):
        M = cv.getPerspectiveTransform(dst, pts)
    else:
        M = cv.getPerspectiveTransform(pts, dst)

    warped = cv.warpPerspective(image, M, (image.shape[1], image.shape[0]))

    return warped


def houghlines(binary_image, draw_image, minimum_votes):
    """ Probablistic hough transform that returns estimated trajectory
    args:
        Binary image
        image to draw lines on
        minimum hough space votes
        
    returns:
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

def getLaneHead(frame, region, res):
    width = frame.shape[1]
    height = frame.shape[0]
    clusters = []
    gap = 0
    for col in range(width):
        column = frame[:,col][height-height//region:]
        sumvalue = np.sum(column)
        if(sumvalue == 0): # 
            gap += 1
        else:
            if gap>res or len(clusters) == 0: # new cluster
                clusters.append([])
            clusters[-1].append(col)
            gap = 0


    return [[sum(x)//len(x), height] for x in clusters]

def getBoundingBox(frame, bottomcenter, w, h, draw=False):
    """ Returns average x position of pixels in bounding box

    """
    cols = range(bottomcenter[0]-w//2, bottomcenter[0]+w//2)
    length = 0
    xpos = 0
    for col in cols:
        column = frame[:,col][bottomcenter[1]-h:bottomcenter[1]]
        sumvalue = np.sum(column)
        length += sumvalue
        xpos += sumvalue*col
    if xpos==0: # if no new center is found, leave it as is
        center = bottomcenter
    else:
        center =  int(xpos//length), bottomcenter[1]

    return center

def drawBoundingBox(frame, bottomcenter, w, h):
    return cv.rectangle(frame, (bottomcenter[0]-w//2, bottomcenter[1]-h), (bottomcenter[0]+w//2, bottomcenter[1]), (0,255,0))
