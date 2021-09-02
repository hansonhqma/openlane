import cv2 as cv
import cvlib as cb
import random

SIZE = (640,360)
NOISE = 5
marker = list((random.randint(0,SIZE[1]), random.randint(0,SIZE[0])))

# PID Parameters

PRO=0.5
INT=0.1
DER=0.4

print(marker)
capture = cv.VideoCapture(0)

def highlightface(frame):
    faces, _ = cb.detect_face(frame)
    centers = []
    if len(faces)!=0:
        for i in range(len(faces)):
            frame = cv.rectangle(frame, (faces[i][0], faces[i][1]), (faces[i][2], faces[i][3]), (0,255,0), 2)
            centers.append((((faces[i][1]+faces[i][3])//2),(faces[i][0]+faces[i][2])//2))
    return frame, tuple(centers)

def find_center_from_mask(mask):
    rowsum, colsum = 0, 0
    r_tagged, c_tagged = 0, 0
    rows, cols = mask.shape[0], mask.shape[1]
    for i in range(rows): # row-wise
        for j in range(cols): # col-wise
            if mask[i][j]==255:
                r_tagged += 1
                c_tagged += 1
                rowsum += i
                colsum += j
    if r_tagged==0:
        return (0,0)
    return (rowsum//r_tagged, colsum//c_tagged)

def draw_xhair(c, frame, color=(0,255,0), t=2):
    return cv.line(cv.line(frame, (c[1],0), (c[1], frame.shape[0]), color,t), (0, c[0]), (frame.shape[1], c[0]), color, t)

x_int_err = 0
y_int_err = 0
x_prev_err = -1
y_prev_err = -1

while True:

    ret, frame = capture.read()
    if not ret:
        break

    # resize and draw target
    frame = cv.resize(frame, SIZE)
    frame, centers = highlightface(frame)
    if len(centers)!=0:
        target = centers[0]
        frame = draw_xhair(target, frame)
    else:
        target = (0,0)


    # add some noise to the marker
    marker[0] = max(0,min(SIZE[1], marker[0]+random.randint(0,NOISE)))
    marker[1] = max(0,min(SIZE[0], marker[1]+random.randint(0,NOISE)))

    frame = draw_xhair((int(marker[0]), int(marker[1])), frame, color=(255,0,255))

    # calculate error
    x_err = target[1]-marker[1]
    y_err = target[0]-marker[0]

    x_int_err += x_err
    y_int_err += y_err

    # calculate step
    if(x_prev_err==-1):
        delta_x = 0
    if(y_prev_err==-1):
        delta_y = 0
    x_step = PRO*x_err + INT*x_int_err + DER*delta_x
    y_step = PRO*y_err + INT*y_int_err + DER*delta_y

    x_prev_err = x_err
    y_prev_err = y_err
    
    # update marker

    marker[0]+=y_step
    marker[1]+=x_step


    cv.imshow('faces', frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break












