import json
from typing import List
import cv2 as cv
import numpy as np
import mss
from ultralytics import YOLO
from datetime import datetime
from re import escape
from flask import Flask
import udp_test as udp
import time

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

def determine_if_q1(box):
    return True if box[0] < frame_points[2][0] and box[0] > frame_points[0][0] and box[1] < frame_points[-1][1] else False

def determine_if_q2(box):
    return True if box[0] < frame_points[2][0] and box[0] > frame_points[0][0] and box[1] > frame_points[-1][1] else False

def determine_if_q3(box):
    return True if box[0] > frame_points[6][0] and box[0] < frame_points[0][0] and box[1] > frame_points[-1][1] else False

def determine_if_q4(box):
    return True if box[0] > frame_points[6][0] and box[0] < frame_points[0][0] and box[1] < frame_points[-1][1] else False

def calculate_ratios(map_pt1, map_pt2, frame_pt1, frame_pt2):
    print("DATA VALUES", map_pt1, map_pt2, frame_pt1, frame_pt2)
    x = abs(map_pt2[0] - map_pt1[0]) / abs(frame_pt2[0] - frame_pt1[0])
    y = abs(map_pt2[1] - map_pt1[1]) / abs(frame_pt2[1] - frame_pt1[1])

    return [x, y]

def distance_away_map_coords(box, ratio_x, ratio_y, reference_point):
    reference_x = reference_point[0]
    box_x = box[0]

    refernce_y = reference_point[1]
    box_y = box[1]
    
    print(box_x, reference_x)
    distance_x = abs(box_x - reference_x)
    distance_y = abs(box_y - refernce_y)

    distance_in_map_x = distance_x * ratio_x
    distance_in_map_y = distance_y * ratio_y
    print(distance_x, distance_in_map_x)
    print(distance_y, distance_in_map_y)
    return [distance_in_map_x, distance_in_map_y]


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

# cap = cv.VideoCapture("./sample.mp4")

target_id = -1
relative_positions = []

number_of_cars : int = 0
distances : List[float] = []
quadrants : List[str] = []
previous_time = time.time()

while True:
    frame = np.array(sct.grab(bounding_box))
    frame = frame[:, :, :3]
    

    # ret, frame = cap.read()

    # if not ret:
    #    print(relative_positions)
    #    break

    result = model.track(frame, persist=True)
    
    annotated_frame = result[0].plot()
    if result[0]:
        if result[0].boxes is not None:
            boxes = result[0].boxes.xywh.cpu().tolist()
            ides = []
            if result[0].boxes.id is not None:
                ides = result[0].boxes.id.int().cpu().tolist()
            
            map_reference_pt1 = []
            map_reference_pt2 = []

            frame_reference_pt1 = []
            frame_reference_pt2 = []
            reference_point = []
            assigned  : bool = False
            quadrant : str = ""

            for box, id in zip(boxes, ides):
                
                if determine_if_q1(box):
                    print("WE'RE IN Q1")
                    map_reference_pt1 = map_points[0]
                    map_reference_pt2 = map_points[2]

                    frame_reference_pt1 = frame_points[0]
                    frame_reference_pt2 = frame_points[2]
                    reference_point = frame_points[0]
                    assigned = True
                    quadrant = "q1"

                if determine_if_q2(box):
                   print("WE'RE IN Q2")
                   map_reference_pt1 = map_points[2]
                   map_reference_pt2 = map_points[4]

                   frame_reference_pt1 = frame_points[2]
                   frame_reference_pt2 = frame_points[4]
                   reference_point = frame_points[2]
                   assigned = True
                   quadrant = "q2"
                    
                if determine_if_q3(box):
                    print("WE'RE IN Q3")
                    map_reference_pt1 = map_points[4]
                    map_reference_pt2 = map_points[6]

                    frame_reference_pt1 = frame_points[4]
                    frame_reference_pt2 = frame_points[6]
                    reference_point = frame_points[4]
                    assigned = True
                    quadrant = "q3"

                if determine_if_q4(box):
                   print("WE'RE IN Q4")
                   map_reference_pt1 = map_points[6]
                   map_reference_pt2 = map_points[8]

                   frame_reference_pt1 = frame_points[6]
                   frame_reference_pt2 = frame_points[8]
                   reference_point = frame_points[6]
                   assigned = True
                   quadrant = "q4"
                
                if assigned:
                    ratio_x, ratio_y = calculate_ratios(
                            map_reference_pt1, 
                            map_reference_pt2, 
                            frame_reference_pt1, 
                            frame_reference_pt2
                    )

                    map_distance = distance_away_map_coords(
                            box, 
                            ratio_x, 
                            ratio_y, 
                            reference_point
                    )
                    
                    cv.putText(
                            annotated_frame, 
                            "This one",
                            (int(box[0]), int(box[1])), 
                            cv.FONT_HERSHEY_COMPLEX, 
                            2, 
                            (0, 0, 255)
                    )
                        
                    relative_positions.append(map_distance)
                    quadrants.append(quadrant)
                    assigned = False 
                    # just sending the raw distance

            # after the for loop completed
            number_of_cars = len(boxes)
            distances = relative_positions
            time = str(datetime.now().time())
                        
            dict = {
                    "number_of_cars": str(number_of_cars),
                    "positions": distances,
                    "quadrants" : quadrants,
                    "time" : time
            }

            relative_positions = []
            quadrants = []

            try:
                MESSAGE = json.dumps(dict)
                udp.send_udp_packet_json_2(MESSAGE)
            except:
                print("something went wrong")
                print(dict)
    for index, point in enumerate(frame_points):
        cv.putText(annotated_frame, "pt"+str(index), (point[0], point[1]), cv.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255))
        cv.circle(annotated_frame, (point[0],point[1]), 10, (255, 0, 0), -1)
        
    cv.imshow("Frame", annotated_frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        print(points)
        print(relative_positions)
        break



