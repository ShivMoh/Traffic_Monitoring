import cv2 as cv
import numpy as np
import mss
from ultralytics import YOLO
from datetime import date, datetime, time

bounding_box = {'top':0, 'left':0, 'width':1920, 'height':1080} 

model = YOLO('/home/shivesh/Documents/python/open_cv/computer_vision/traffic/runs/detect/train/weights/best.pt')
sct = mss.mss()

# center, (start from to, then clockise onwoard) pt1, pt2, pt3, pt4
points = []
border_points = [(562, 437), (419, 376), (1395, 430), (1168, 768), (10, 10)]

def draw_circle(event,x,y,flags,param):
    global annotated_frame, points  
    if event == cv.EVENT_LBUTTONDOWN:
        points.append((x,y))
        cv.circle(annotated_frame, (x,y), 5, (255, 0, 0), -1)
 
annotated_frame = np.zeros((1920, 1080)) 
cv.namedWindow('Frame')
cv.setMouseCallback('Frame',draw_circle)
 
while True:
    frame = np.array(sct.grab(bounding_box))
    frame = frame[:, :, :3]
    result = model.track(frame, persist=True)

    annotated_frame = result[0].plot()
            
    for index, point in enumerate(border_points):
        cv.putText(annotated_frame, "pt"+str(index), (point[0], point[1]), cv.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255))
        cv.circle(annotated_frame, (point[0],point[1]), 10, (255, 0, 0), -1)
        
    cv.imshow("Frame", annotated_frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        print(points)
        break
