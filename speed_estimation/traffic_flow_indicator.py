from typing import List
import cv2 as cv
from ultralytics import YOLO, solutions
import mss
import numpy as np
import math
import datetime
import estimator_funcs as utils

sct = mss.mss()

bounding_box = {'top':0, 'left':0, 'width':1920, 'height':1080} 
model = YOLO('/home/shivesh/Documents/python/open_cv/computer_vision/traffic/runs/detect/train/weights/best.pt')
names = model.model.names

# fill empty list with first batch of boxes and ids
# then second frame check if there are common ids between the previous ids and the current names
# if yes, calculate the distance travelled and time taken inorder to calculate the speed pixels/second
#   discard those that did not repeat --> eh...something to consider ig
# update current ids
# update boxes

# store last know coordinates with time

# if point is difference by say n amount then
# we'll calculate distance traveled 

def parse_objs(boxes : List[List[float]]) -> List[dict]:
    """
        This function will parse all the boxes into the required format
    """
    ret_list = [] 
    for box in boxes:
        obj = {
            "coordinate": [box[0], box[1]],
            "time_stamp": utils.get_current_time()
        }
        ret_list.append(obj)

    return ret_list

def calculate_average_speed(previous_boxes : List[dict], current_boxes : List[dict], previous_ids : List[int], current_ids : List[int]) -> float:
    count = 0
    total_estimated_speed = 0
    
    for current_index, id in enumerate(current_ids):

        if id in previous_ids:
            # calculate the speed
            previous_index = previous_ids.index(id)
            previous_box = previous_boxes[previous_index]
            current_box = current_boxes[current_index] 

            distance_travelled = math.sqrt((current_box["coordinate"][0] - previous_box["coordinate"][0]) ** 2 + (current_box["coordinate"][1] - previous_box["coordinate"][1]) ** 2)
            
            time_taken = str(utils.str2time(current_box["time_stamp"]) - utils.str2time(previous_box["time_stamp"]))

            time_taken = utils.time2sec(utils.str2time(time_taken))
            # pixels / second
            estimated_speed = distance_travelled / time_taken
            count += 1
            total_estimated_speed += estimated_speed

    if count > 0 and total_estimated_speed > 0: 
        average_estimated_speed = total_estimated_speed / count
        return average_estimated_speed
    
    return 0.0

previous_boxes = None
previous_ides = None
current_average = 0

while True:
    raw_frame = np.array(sct.grab(bounding_box))
    
    frame = raw_frame[:, :, :3]
    result = model.track(frame, persist=True, show=False)
     
    if result:
        if result[0].boxes is not None and result[0].boxes.id is not None:
            boxes = result[0].boxes.xywh.cpu().tolist()
            ides = result[0].boxes.id.int().cpu().tolist()
            
            # calculate new distance travelled (pixels)
            if previous_boxes is not None and previous_ides is not None:
                current_boxes = parse_objs(boxes)
                average_speed = calculate_average_speed(previous_boxes=previous_boxes, current_boxes=current_boxes, previous_ids=previous_ides, current_ids=ides)

                if average_speed > 0.0:
                    current_average = (average_speed + current_average) / 2
                    
            previous_boxes = parse_objs(boxes)
            previous_ides = ides

    cv.putText(raw_frame, "Average speed is (pixels/second)" + str(current_average), (50,50), fontFace=cv.FONT_HERSHEY_COMPLEX, fontScale=1, color=(255, 0, 0)); 

    cv.imshow("Speed Indicator Frame", raw_frame)

    if cv.waitKey(25) & 0xFF == ord('q'):
       break 

cv.destroyAllWindows()
