from ultralytics import YOLO
import cv2 as cv
import mss
import numpy as np
from datetime import datetime


bounding_box = {'top':0, 'left':0, 'width':1920, 'height':1080} 


model = YOLO('/home/shivesh/Documents/python/open_cv/computer_vision/traffic/runs/detect/train/weights/best.pt')
sct = mss.mss()
ret = True
max_count = 0

boundary_1 = [(330, 380), (330, 450)]
boundary_2 = [(1030, 380), (1030, 450)]

time_store = {
   "time_stamp_b1" : "",
    "time_stamp_b2" : "",
    "passed_both_boundaries" : False
}

time_stores = []
passed_check_point_1 = []
passed_check_point_2 = []
def check_if_pass_first_boundary(box, id):
   
   if id in passed_check_point_1:
       return True

   is_y1_within_bounds = False
   is_y2_within_bounds = True
   is_x1_passed = False
     
   if box[1] > float( boundary_1[0][1]):
       is_y1_within_bounds = True

 #  if box[1] + box[3] < float(boundary_1[1][1]):
 #      is_y2_within_bounds = True

   if box[0] > float(boundary_1[0][0]):
       is_x1_passed = True
   
   print("BOX:", box)
   print("ID", id)
   print("y1", is_y1_within_bounds)
   print("y2", is_y2_within_bounds)
   print("x1", is_x1_passed)

   print("-------------------")
   
   return_bool = is_y1_within_bounds and is_y2_within_bounds and is_x1_passed
   
   if return_bool: 
      current_time = datetime.now()
      current_time = current_time.strftime("%H:%M:%S.%f")
      data_point = [id, current_time, "", False]
      time_stores.append(data_point)
      passed_check_point_1.append(id)

   return return_bool

def check_if_pass_second_boundary(box, id):
    if id in passed_check_point_2:
        return True

    is_x2_passed = False

    if box[0] > float(boundary_2[0][0]):
        is_x2_passed = True
    return_bool = is_x2_passed

    if return_bool:
       index_to_change = -1
       for index, data_point in enumerate(time_stores):
           if id in passed_check_point_1: 
               print("YES", id)
           if data_point[0] == id:
              index_to_change = index
              break
       
       if index_to_change != -1:
           current_time = datetime.now()
           current_time = current_time.strftime("%H:%M:%S.%f")
           time_stores[index_to_change][2] = current_time
           time_stores[index_to_change][3] = True
           passed_check_point_2.append(id) 

    return return_bool

while ret:
    # ret, frame = cap.read()
    ret = True
    raw_frame = np.array(sct.grab(bounding_box))
    frame = raw_frame[:, :, :3]

    if ret:
        results = model.track(frame, persist=True)
        
        if (results):
            ids = results[0].boxes.id

            if ids is not None:
                boxes = results[0].boxes.xywh.cpu().tolist()
                ides = ids.int().cpu().tolist()

                if not (len(boxes) == len(ides)):
                    print("NOT EQUAL")          

                for index, box in enumerate(boxes):
                    check_if_pass_first_boundary(box, ides[index])
                    check_if_pass_second_boundary(box, ides[index])
                
                count = max(ids.int().cpu().tolist())
                max_count = count

        frame_ = results[0].plot()

        cv.line(frame_,boundary_1[0], boundary_1[1], color=(0, 255, 0), thickness=10)        
        cv.line(frame_, boundary_2[0], boundary_2[1], color=(0, 255, 0), thickness=10)        

        frame_ = cv.resize(frame_, (int(frame_.shape[1] * 0.5), int(frame_.shape[0] * 0.5)), interpolation=cv.INTER_AREA)  
        

        cv.imshow('frame', frame_)
        if cv.waitKey(25) & 0xFF == ord('q'):
            break

print(time_stores)
print("Final Count", max_count)
