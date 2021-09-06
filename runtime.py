from pathlibcv import *

box_count = 8

capture = cv.VideoCapture("test.mp4")
firstframe = True

laneHeads = []

while(True):
    ret, frame = capture.read()
    if not ret:
        break
    
    pts = np.array(([575,564],[680,564],[744,620],[490,620])).astype("float32") # corners of square on surface
    mask_corners = np.array([[570, 540], [680,540], [900, 681], [300, 683]]) # corners of mask
    pts, mask_corners = pts//2, mask_corners//2 # resize corners for resized frame
    hsv_min = (0,0,142)
    hsv_max = (180,255,255)

    frame = fastresize(frame, 0.5) # resize frame for faster processing

    mask = drawMask(frame, mask_corners) # draw mask on original image
    transformed = squarePerspectiveTransform(mask, pts) # perform square transform
    binary_image = hsvThreshold(transformed, hsv_min, hsv_max) # hsv thresholding to get binary image
 
    frame = cv.polylines(frame, [mask_corners.reshape((-1,1,2))], True, (0,255,0), 1) # mark mask on original image

    if firstframe: # first pass, used getLaneHeads
        firstframe = False
        laneHeads = getLaneHead(binary_image, box_count, 10)


    binary_image = cv.cvtColor(binary_image, cv.COLOR_GRAY2BGR)

    for head in laneHeads:
        binary_image = cv.line(binary_image, (head, binary_image.shape[0]), (head, 0), (0,255,0))

        
    

    # display
    cv.imshow("original", frame)
    cv.imshow("binary_image", binary_image)


    if cv.waitKey(1) & 0xFF==ord('q'):
        break
