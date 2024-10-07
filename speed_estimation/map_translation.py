import cv2 as cv
from cv2.gapi import parseSSD
import numpy as np
import mss
from ultralytics import YOLO
from datetime import date, datetime, time
from re import escape
from flask import Flask

bounding_box = {'top':0, 'left':0, 'width':1920, 'height':1080} 

model = YOLO('/home/shivesh/Documents/python/open_cv/computer_vision/traffic/runs/detect/train/weights/best.pt')
sct = mss.mss()

map_points = [
	(6.777571183264428, -58.19654292167958),
	(6.777443999966449, -58.19648986995),
	(6.777316816702022, -58.196542921747145),
	(6.777264135678817, -58.196670999999995),
	(6.777316816702022, -58.19679907825285),
	(6.777443999966449, -58.19685213005),
	(6.777571183264428, -58.19679907832042),
	(6.777623864321185, -58.196670999999995),
	(6.777444, -58.196671)
]

frame_points = [
(714, 268), (1193, 268), (1654, 333), (1770, 455), (1426, 618), (261, 681), (2, 380), (424, 288), (829, 324)

]

# center, (start from to, then clockise onwoard) pt1, pt2, pt3, pt4
points = []

border_points = [
        (562, 437), 
        (419, 376), 
        (1395, 430), (1168, 768), (10, 10)]

# pt1, pt3, pt5, pt7
# [(751, 259), (1658, 333), (1426, 667), (1, 415)]

# pt2, pt4, pt6, pt8
# [(1234, 276), (1836, 467), (235, 670), (391, 287)]

def determine_quadrant(box):
    return True if box[0] < frame_points[2][0] and box[0] > frame_points[0][0] and box[1] < frame_points[-1][1] else False

def calculate_ratio(map_pt1, map_pt2, frame_pt1, frame_pt2):
    return (map_pt2 - map_pt1) / (frame_pt2 - frame_pt1)

def distance_away_map_coords(box, ratio, reference_point):
    reference_x = reference_point[0]
    box_x = box[0]

    distance = box_x - reference_x

    distance_in_map = distance * ratio

    return distance_in_map
app = Flask(__name__)


# frame coords
"""
    pt1 : (751, 259),
    pt2 : (1234, 276)
    pt3 : (1658, 333)
    pt4 : (1836, 467)
    pt5 : (1426, 667)
    pt6 : (235, 670)
    pt7 : (1, 415)
    pt8 : (391, 287)
    coords : (562, 437)
"""

# map coords
"""
	pt1 : [6.777571183264428, -58.19654292167958],
	pt2 : [6.777443999966449, -58.19648986995],
	pt3 : [6.777316816702022, -58.196542921747145],
	pt4 : [6.777264135678817, -58.196670999999995],
	pt5 : [6.777316816702022, -58.19679907825285],
	pt6 : [6.777443999966449, -58.19685213005],
	pt7 : [6.777571183264428, -58.19679907832042],
	pt8 : [6.777623864321185, -58.196670999999995],
	center : [6.777444, -58.196671]
"""

def draw_circle(event,x,y,flags,param):
    global annotated_frame, points  
    if event == cv.EVENT_LBUTTONDOWN:
        points.append((x,y))
        cv.circle(annotated_frame, (x,y), 5, (255, 0, 0), -1)
 
annotated_frame = np.zeros((1920, 1080)) 
cv.namedWindow('Frame')
cv.setMouseCallback('Frame',draw_circle)

cap = cv.VideoCapture("./sample.mp4")

target_id = 20
relative_positions = []

while True:
    # frame = np.array(sct.grab(bounding_box))
    # frame = frame[:, :, :3]

    ret, frame = cap.read()
    result = model.track(frame, persist=True)
    
    annotated_frame = result[0].plot()
    if result[0]:
        if result[0].boxes is not None:
            boxes = result[0].boxes.xywh.cpu().tolist()
            ides = []
            if result[0].boxes.id is not None:
                ides = result[0].boxes.id.int().cpu().tolist()

            for box, id in zip(boxes, ides):
                if determine_quadrant(box):
                    if target_id == -1:
                        target_id = id
                    if id == target_id:
                        print("We're in Q1")
                        ratio = calculate_ratio(map_points[0][0], map_points[2][0], frame_points[0][0], frame_points[2][0])
                        map_distance = distance_away_map_coords(box, ratio, frame_points[0])
                        
                        print(map_distance)
                        cv.putText(annotated_frame, "This one", (int(box[0]), int(box[1])), cv.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255))
                        relative_positions.append(map_distance)

            
    for index, point in enumerate(frame_points):
        cv.putText(annotated_frame, "pt"+str(index), (point[0], point[1]), cv.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255))
        cv.circle(annotated_frame, (point[0],point[1]), 10, (255, 0, 0), -1)
        
    cv.imshow("Frame", annotated_frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        print(points)
        print(relative_positions)
        break



