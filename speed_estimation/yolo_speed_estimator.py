from ultralytics import YOLO
import cv2 as cv
import mss
import numpy as np
from datetime import date, datetime, time


bounding_box = {'top':0, 'left':0, 'width':1920, 'height':1080} 


model = YOLO('/home/shivesh/Documents/python/open_cv/computer_vision/traffic/runs/detect/train/weights/best.pt')
sct = mss.mss()
ret = True
max_count = 0

boundary_1 = [(330, 380), (330, 450)]
boundary_2 = [(1030, 380), (1030, 450)]
first_boundary_pass = []

# id, t1, t2, passed_both_boundaries
time_stores = []
passed_check_point_1 = []
passed_check_point_2 = []

def check_both_over(box):
    
    is_y1_within_bounds_b1 = False
    is_x1_passed_b1 = False
     
    if box[1] > float( boundary_1[0][1]):
       is_y1_within_bounds_b1 = True

    if box[0] > float(boundary_1[0][0]):
       is_x1_passed_b1 = True


    is_x2_passed_b2 = False

    if box[0] > float(boundary_2[0][0]):
        is_x2_passed_b2 = True

    return is_y1_within_bounds_b1 and is_x1_passed_b1 and is_x2_passed_b2 

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
   
#   print("BOX:", box)
#   print("ID", id)
#   print("y1", is_y1_within_bounds)
#   print("y2", is_y2_within_bounds)
#   print("x1", is_x1_passed)

#   print("-------------------")
   
   return_bool = is_y1_within_bounds and is_y2_within_bounds and is_x1_passed
   
   if return_bool: 
      current_time = datetime.now()
      current_time = current_time.strftime("%H:%M:%S.%f")
      data_point = [id, current_time, "", False]
      time_stores.append(data_point)
      passed_check_point_1.append(id)
      first_boundary_pass.append(id)

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

def get_where_passed_both_boundaries(time_stores):
    ret_list = [] 
    for list in time_stores:
        if list[3]:
           ret_list.append(list) 

    return ret_list

def passed_both_boundaries(id):
    for data in time_stores:
        if data[0] == id:
            return data
    return [False, False, False, False]

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
                    if ides[index] not in first_boundary_pass:
                        both_over = check_both_over(box) 

                        if not both_over:
                            print("Not Both Over")
                            check_if_pass_first_boundary(box, ides[index])

                    if ides[index] in first_boundary_pass:
                        check_if_pass_second_boundary(box, ides[index])
                    
                    data = passed_both_boundaries(ides[index]) 

                    if data[3] is True:

                        x_1 = int(box[0] - box[2] / 2)
                        x_2 = int(x_1 + box[2])

                        y_1 = int(box[1] - box[3] / 2)
                        y_2 = int(y_1 + box[3])

                        distance = 28 / 1000
                        time_taken = datetime.strptime(data[2], "%H:%M:%S.%f") - datetime.strptime(data[1], "%H:%M:%S.%f")
                        time_taken = float(datetime.strftime(datetime.strptime(str(time_taken),"%H:%M:%S.%f"), "%S.%f"))

                        print("time taken, id", time_taken, id)
                        speed = distance / time_taken * 1/25

                        cv.putText(raw_frame, str(speed), (x_1, y_1), cv.FONT_HERSHEY_COMPLEX, 3, (0, 0, 255))
                        cv.rectangle(raw_frame, (x_1, y_1), (x_2, y_2), (255, 0, 0), 2)

                    
                
                count = max(ids.int().cpu().tolist())
                max_count = count

        frame_ = results[0].plot()

        cv.line(frame_,boundary_1[0], boundary_1[1], color=(0, 255, 0), thickness=10)        
        cv.line(frame_, boundary_2[0], boundary_2[1], color=(0, 255, 0), thickness=10)        

        frame_ = cv.resize(frame_, (int(frame_.shape[1] * 0.5), int(frame_.shape[0] * 0.5)), interpolation=cv.INTER_AREA)  
        

        cv.imshow('Raw_Frame', raw_frame)
        if cv.waitKey(25) & 0xFF == ord('q'):
            break


new_time_stores = get_where_passed_both_boundaries(time_stores)

for data_point in new_time_stores:
    first_data_point = data_point
    distance = 28 / 1000
    time_taken = datetime.strptime(first_data_point[2], "%H:%M:%S.%f") - datetime.strptime(first_data_point[1], "%H:%M:%S.%f")
    time_taken = float(datetime.strftime(datetime.strptime(str(time_taken),"%H:%M:%S.%f"), "%S.%f"))


    speed = distance / time_taken * 1/25
    print(distance)
    print(time_taken)
    print(speed)

# print ("Filtered List", time_stores)
# print("Final Count", max_count)
