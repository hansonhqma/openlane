import sys
from pathlibcv import *
import sys
from collections import deque
import time

framerate = deque(maxlen=50)

box_count = 6
box_width = 50
frame_scale = 2

capture = cv.VideoCapture("driving.mp4")
firstframe = True

<<<<<<< HEAD
hsv_min = (0,0,161)
hsv_max = (180,255,255)

mask_corners = np.array(([563,510],[700,510],
    [1280,720],[0,720])) # corners of square on surface
pts = np.array([[585, 527], [695,527],
    [730, 554], [540, 554]]) # corners of mask
=======
pts = np.array(([575,564],[680,564],[744,620],[490,620])) # corners of square on surface
mask_corners = np.array([[530, 540], [710,540], [930, 683], [270, 683]]) # corners of mask
>>>>>>> 0e81e65ef1140c0999fba45f5ca0e65d49f56505
pts, mask_corners = pts//frame_scale, mask_corners//frame_scale # resize corners for resized frame
pts, mask_corners = pts.astype('int64'), mask_corners.astype('int64')

marker_color = (0,0,255)
marker_size = 5
boxes = []

<<<<<<< HEAD
showmask = 'showmask' in sys.argv
=======

showmask = False
if 'showmask' in sys.argv:
    showmask = True
>>>>>>> 0e81e65ef1140c0999fba45f5ca0e65d49f56505

while(True):
    NS_TIME = time.clock_gettime_ns(time.CLOCK_REALTIME)
    ret, frame = capture.read()
    if not ret:
        break

    frame = fastresize(frame, 1/frame_scale) # resize frame for faster processing
    FRAME_HEIGHT = frame.shape[0]
    FRAME_WIDTH = frame.shape[1]
    box_height = FRAME_HEIGHT//box_count


    mask = drawMask(frame, mask_corners) # draw mask on original image
    if showmask:
        frame = cv.polylines(frame, [mask_corners], True, (0,255,0))
    transformed = squarePerspectiveTransform(mask, pts) # perform square transform
    binary_image = hsvThreshold(transformed, hsv_min, hsv_max) # hsv thresholding to get binary image
 
    lane_markers = cv.cvtColor(np.zeros((FRAME_HEIGHT, FRAME_WIDTH)).astype("uint8"), cv.COLOR_GRAY2BGR)

    if firstframe: # first pass, get lane starts, then build the box positions
        firstframe = False

        lane_start_positions = getLaneHead(binary_image, box_count, 5) # get coordinates of lane starts
        for lane in lane_start_positions:
            boxes.append([lane for x in range(box_count)]) # fill with start position
    
    for lane in boxes:
        lane[0] = getBoundingBox(binary_image, lane[0], box_width, box_height) # update bottom first
        lane_markers = cv.circle(lane_markers, lane[0], marker_size, marker_color, -1)
        transformed = drawBoundingBox(transformed, lane[0], box_width, box_height)
        for i in range(1,len(lane)):
            bottom_center = [lane[i-1][0], lane[i-1][1]-box_height] # calculate next box pos based on previous box
            lane[i] = getBoundingBox(binary_image, bottom_center, box_width, box_height)
            lane_markers = cv.circle(lane_markers, lane[i], marker_size, marker_color, -1)
            transformed = drawBoundingBox(transformed, lane[i], box_width, box_height)

    lane_markers = squarePerspectiveTransform(lane_markers, pts, reverse=True)
    frame = cv.addWeighted(frame, 1, lane_markers, 1, 0)
        
    if showmask:
        frame = cv.polylines(frame, [mask_corners], True, (0,255,255))
        frame = cv.polylines(frame, [pts], True, (255,255,0))
    
    # upscale and display
    cv.imshow("original", frame)
    cv.imshow("binary_image", binary_image)
<<<<<<< HEAD
    cv.imshow("transformed", transformed)
=======
>>>>>>> 0e81e65ef1140c0999fba45f5ca0e65d49f56505

    TIME_DELTA = (time.clock_gettime_ns(time.CLOCK_REALTIME)-NS_TIME)/1000000000
    framerate.append(1/TIME_DELTA)


    if cv.waitKey(1) & 0xFF==ord('q'):
        break

print("Average fps: {:.2f}".format(sum(framerate)/len(framerate)))

