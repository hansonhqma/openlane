from pathlibcv import *
import time

box_count = 8
box_width = 50
frame_scale = 3

capture = cv.VideoCapture("test.mp4")
firstframe = True

pts = np.array(([575,564],[680,564],[744,620],[490,620])).astype("float32") # corners of square on surface
mask_corners = np.array([[570, 540], [680,540], [900, 681], [300, 683]]) # corners of mask
pts, mask_corners = pts//frame_scale, mask_corners//frame_scale # resize corners for resized frame

boxes = []

while(True):
    NS_TIME = time.clock_gettime_ns(time.CLOCK_REALTIME)
    ret, frame = capture.read()
    if not ret:
        break
    
    hsv_min = (0,0,142)
    hsv_max = (180,255,255)

    frame = fastresize(frame, 1/frame_scale) # resize frame for faster processing
    FRAME_HEIGHT = frame.shape[0]
    FRAME_WIDTH = frame.shape[1]
    box_height = FRAME_HEIGHT//box_count

    mask = drawMask(frame, mask_corners) # draw mask on original image
    transformed = squarePerspectiveTransform(mask, pts) # perform square transform
    binary_image = hsvThreshold(transformed, hsv_min, hsv_max) # hsv thresholding to get binary image
 
    frame = cv.polylines(frame, [mask_corners.reshape((-1,1,2))], True, (0,255,0), 1) # mark mask on original image

    if firstframe: # first pass, get lane starts, then build the box positions
        firstframe = False

        lane_start_positions = getLaneHead(binary_image, box_count, 5)
        for lane_pos in lane_start_positions:
            # get first box
            lane_boxes = [[lane_pos, FRAME_HEIGHT]]
            
            for box_no in range(1,box_count):
                box_pos_x = lane_boxes[-1][0]
                box_pos_y = lane_boxes[-1][1]-box_height
                update = getBoundingbox(binary_image, (box_pos_x, box_pos_y), box_width, box_height)
                lane_boxes.append([update[0], update[1]]) # add box pos
            boxes.append(lane_boxes)
    else: # update boxes
        for lane in boxes:
            for i in range(len(lane)):
                update = getBoundingbox(binary_image, lane[i], box_width, box_height)
                lane[i][0] = update[0]
                lane[i][1] = update[1]
                transformed = drawBoundingBox(transformed, lane[i], box_width, box_height)

    TIME_DELTA = (time.clock_gettime_ns(time.CLOCK_REALTIME)-NS_TIME)/1000000000
    print("FPS: {:.2f}".format(1/TIME_DELTA))
    
        

    # display
    cv.imshow("original", frame)
    cv.imshow("transformed", transformed)
    cv.imshow("binary_image", binary_image)


    if cv.waitKey(1) & 0xFF==ord('q'):
        break
