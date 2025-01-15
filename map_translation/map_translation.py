import json 
from typing import List
import cv2 as cv
import numpy as np
import mss
from ultralytics import YOLO
from datetime import datetime
import udp_test as udp
from constants import MODEL_PATH, FRAME_POINTS, MAP_POINTS, BOUNDING_BOX
from utils import get_data_for_quadrant, calculate_ratios, distance_away_map_coords
import time

model = YOLO(MODEL_PATH)
points = []
annotated_frame = np.zeros((1920, 1080)) 
target_id = -1
relative_positions = []
number_of_cars : int = 0 distances : List[float] = []
quadrants : List[str] = []
previous_time = time.time()
sct = mss.mss()

while True:
    frame = np.array(sct.grab(BOUNDING_BOX))
    frame = frame[:, :, :3]
    result = model.track(frame, persist=True)
    annotated_frame = result[0].plot()

    if result[0]:
        if result[0].boxes is not None:
            boxes = result[0].boxes.xywh.cpu().tolist()
            ides = []
            if result[0].boxes.id is not None:
                ides = result[0].boxes.id.int().cpu().tolist()
            
            for box, id in zip(boxes, ides):
                map_reference_pt1, map_reference_pt2, frame_reference_pt1, frame_reference_pt2, reference_point, assigned, quadrant = get_data_for_quadrant(box) 

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

    for index, point in enumerate(FRAME_POINTS):
        cv.putText(annotated_frame, "pt"+str(index), (point[0], point[1]), cv.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255))
        cv.circle(annotated_frame, (point[0],point[1]), 10, (255, 0, 0), -1)
        
    cv.imshow("Frame", annotated_frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break



