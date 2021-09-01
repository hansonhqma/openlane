import cv2
import numpy as np


# ----------------- IMAGE PREPROCESSING ------------------------
def preprocess(img, resize=True, bilaterial=True, gaussian=True,
               IMG_WIDTH=640, IMG_HEIGHT=480,
               bilatDiameter=9, bilatSigmaColor=150, bilatSigmaSpace=150,
               gaussKernelSize=3):

    """Preprocess image for contours (resize, gaussian blur, etc)"""

    # Make sure we don't waste time resizing an image to the same size
    if ((img.shape[0] != IMG_WIDTH) or (img.shape[1] != IMG_HEIGHT)) and resize:
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT), cv2.INTER_AREA)

    if bilaterial:  # Bilaterial blur, removes noise but is slow af
        img = cv2.bilateralFilter(
            img, bilatDiameter, bilatSigmaColor, bilatSigmaSpace
        )

    if gaussian:  # Gaussian blur
        img = cv2.GaussianBlur(
            img, (gaussKernelSize, gaussKernelSize), 0
        )

    return img


# --------------------- HSL/HSV -----------------------------
def hsv(img, lower, upper):
    """Takes image and H, S, and V values, returns binary image"""
    assert type(lower) is tuple  # In case we try to passa non-tuple
    assert type(upper) is tuple
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img = cv2.inRange(img, lower, upper)
    return img


# ---------------- BINARY IMAGE PROCESSING -------------------
def imgFix(img, kernelSize=4):
    """Adds the closed and opened versions of a binary image to (theoretically)
    fix many defects that occur after HSL/HSV masking"""
    kernel = np.ones((kernelSize, kernelSize), np.uint8)
    open_img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    close_img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    return cv2.add(open_img, close_img)


def dilate(img, kernelSize):
    """ Dilates image"""
    kernel = np.ones((kernelSize, kernelSize), np.uint8)
    return cv2.morphologyEx(img, cv2.MORPH_DILATE, kernel)


def erode(img, kernelSize):
    kernel = np.ones((kernelSize, kernelSize), np.uint8)
    return cv2.morphologyEx(img, cv2.MORPH_ERODE, kernel)


# --------------- FINDING CONTOURS --------------------------
def findContours(img):
    """Finds all contours in an image. Real exciting stuff"""
    ret, contours, hierarchy = \
        cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours
    # Note: Change cv2.RETR_EXTERNAL to cv2.RETR_TREE if we're having issues
    # with bad or spotty contour detection


# -------------- FILTERING AND MANIPULATING CONTOURS -----------------
def heightRange(contourList, low, high, left, right):
    """Only keeps contours that are 100% between low and high pixels"""
    newList = []
    for i in contourList:
        x, y, w, h, = cv2.boundingRect(i)
        if (left <= x) and (x + w <= right) and (low <= y) and (y + h <= high):
            newList.append(i)
    return newList


def sizeRange(contourList, low, high):
    """Only keeps contours that are in range for size"""
    newList = []
    for i in contourList:
        if (low <= cv2.contourArea(i) <= high):
            newList.append(i)
    return newList


def aspectRange(contourList, low, high):
    """Only keeps contours that are in range for size"""
    newList = []
    for i in contourList:
        if (low <= getAspect(i) <= high):
            newList.append(i)
    return newList


def convexify(contour):
    """Turns a contour into a convex one"""
    return cv2.convexHull(contour)


def simplify(contour, numPoints, tick=None):
    """Simplifies contour to numPoints total points"""
    if tick is None:
        tick = cv2.arcLength(contour, True)*0.01
    epsilon = cv2.arcLength(contour, True)*0.01
    while len(cv2.approxPolyDP(contour, epsilon, True)) > numPoints:
        epsilon += tick
    return cv2.approxPolyDP(contour, epsilon, True)


# ------------------- CONTOUR INFO/MISC ----------------------
def getSize(contour):
    """Returns contour size"""
    return cv2.contourArea(contour)


def getCoords(contour):
    """Returns contour XY coords"""
    M = cv2.moments(contour)
    x = int(M['m10']/M['m00'])
    y = int(M['m01']/M['m00'])
    return (x, y)


def getAspect(contour):
    """ Returns contour aspect ratio"""
    x, y, w, h = cv2.boundingRect(contour)
    return float(w)/h


def getBoundingBox(contour):
    """ Returns minimum bounding box"""
    return cv2.boundingRect(contour)


def getRotatedBoundingBox(contour):
    """ Returns minimum bounding box, rotated as necessary"""
    return cv2.minAreaRect(contour)


# ------------------- MISC UTILS ----------------------------
def showUntilQ(img):
    """Shows image until q key is pressed"""
    while True:
        cv2.imshow('window', img)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()


def rotatedPointsToRect(points):
    """Turns points in (x1, y1), (x2, y2), r to (x1, y1), (x2, y2)"""
    return cv2.boxPoints(points)
