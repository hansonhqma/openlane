import cv2 as cv
import numpy as np
import os

def fovmask(img, pts):
    """ Isolates field of view to the array of points given by setting all other points to black
    Args: 
        img(numpy.ndarrray): image to be masked
        pts(numpy.ndarrray): array of points describing the fov

    Returns:
        numpy.ndarrray: masked image
    """
    mask = cv.fillPoly(np.zeros(img.shape).astype(np.uint8), pts=[pts], color=(255,255,255)).astype(np.uint8)
    return cv.bitwise_and(img, mask)

def fastresize(img, scaling):
    """ Simple resize function
    Args:
        img(numpy.ndarrray): image to be resized
        scaling(float): postitive scaling factor

    Returns:
        np.array: resized image
    """
    if not scaling > 0:
        raise ValueError("Scaling factor has to be float greater than 0")
    return cv.resize(img, (int(img.shape[1]*scaling), int(img.shape[0]*scaling)))

def undistortFrame(img, cMat, ncMat, dist, roi):
    """ Applies camera matrices to undistort an image, and crops the image correctly
    Args:
        img(numpy.ndarrray): image to be undistorted
        cMat(numpy.ndarrray): camera matrix
        ncMat(numpy.ndarrray): new camera matrix
        dist(numpy.ndarrray): distortion coefficients
        roi(tuple): all-good-pixels region of interest

    Returns:
        numpy.ndarray: undistorted image

    """
    dst = cv.undistort(img, cMat, dist, None, ncMat)
    x,y,w,h = roi
    dst = dst[y:y+h,x:x+w]
    return dst

def getCameraMatrices(folderpath, chessboardSize):
    """ Returns original and optimized camera matrices, dist, and roi used for image undistortion
    Args:
        folderpath(String): filepath to folder containing calibration images
        chessboardSize(tuple): dimensions of chess board by internal corners, cols x rows
    Returns:
        numpy.ndarray: camera matrix
        numpy.ndarray: new camera matrix
        numpy.ndarray: disrotion coefficients
        tuple: all-good-pixels region of interest
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


def squarePerspectiveTransform(img, pts, vshift, SCALING=1, reverse=False):
    """ Performs a 4 point perspective warp on an image
    Args:
        img(numpy.ndarray): source image
        pts(numpy.ndarray): 4-point polygon marking out a square on target surface
        vshift(int): number of pixels to vertically shift the transform
        reverse(boolean): applies the inverse transform if True
    
    Returns:
        numpy.ndarray: transformed image


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

    bounding_box_center = ((x_max+x_min)//2, (y_max+y_min)//2+vshift)

    dst = np.array([[bounding_box_center[0]-square_edge_distance//2, bounding_box_center[1]-square_edge_distance//2],
        [bounding_box_center[0]+square_edge_distance//2, bounding_box_center[1]-square_edge_distance//2],
        [bounding_box_center[0]+square_edge_distance//2, bounding_box_center[1]+square_edge_distance//2],
        [bounding_box_center[0]-square_edge_distance//2, bounding_box_center[1]+square_edge_distance//2]]).astype(np.float32)

    if(reverse):
        M = cv.getPerspectiveTransform(dst, pts)
    else:
        M = cv.getPerspectiveTransform(pts, dst)

    warped = cv.warpPerspective(img, M, (img.shape[1], img.shape[0]))

    return warped


def houghlines(binary_image, minimum_votes):
    """ Probablistic hough transform that Returns estimated trajectory
    Args:
        binary_image(numpy.ndarray): binary image
        minimum_votes(int): minimum hough space votes
        
    Returns:
        float: average angular trajectory
        int: lowest x coordinate
        list: list of lines found
    """
    lines = cv.HoughLinesP(binary_image, 1, np.pi / 180, minimum_votes, None, 50, 10)
    ret = []
    if lines is not None:
        angular_deviation = []
        lowest_point = []
        lowest_y = 0 # image y is flipped, so visually lowest is max value
        for i in range(len(lines)):
            l = lines[i][0]
            ret.append(l)
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
        return sum(angular_deviation)/len(angular_deviation), lowest_point, ret

def binaryImageHistogram(img, region, res):
    """ Takes the histogram of the lower 1/nth region of the image, and extracts clusters by separation distance
    Args:
        img(numpy.ndarray): binary image input
        region(int): n in bottom 1/nth region of the image to be parsed
        res(int): maximum separation between non-zero columns for them to be declared in the same cluster
    Returns:
       list: image coordinates describing estimated start position of each lane 
    """

    width = img.shape[1]
    height = img.shape[0]
    clusters = []
    gap = 0
    for col in range(width):
        column = img[:,col][height-height//region:]
        sumvalue = np.sum(column)
        if(sumvalue == 0): # 
            gap += 1
        else:
            if gap>res or len(clusters) == 0: # new cluster
                clusters.append([])
            clusters[-1].append(col)
            gap = 0


    return [[sum(x)//len(x), height] for x in clusters]

def getBoundingBox(img, bottomcenter, w, h):
    """ Returns average x position of pixels in bounding box
    Args:
        img(numpy.ndarray): binary image input
        bottomcenter(tuple): bottom center coordinate of bounding box
        w(int): bounding box pixel width
        h(int): bounding box pixel height

    """
    cols = range(bottomcenter[0]-w//2, bottomcenter[0]+w//2)
    length = 0
    xpos = 0
    for col in cols:
        column = img[:,col][bottomcenter[1]-h:bottomcenter[1]]
        sumvalue = np.sum(column)
        length += sumvalue
        xpos += sumvalue*col
    if xpos==0: # if no new center is found, leave it as is
        center = bottomcenter
    else:
        center =  int(xpos//length), bottomcenter[1]

    return center
