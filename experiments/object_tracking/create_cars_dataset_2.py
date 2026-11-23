from ultralytics import YOLO
import cv2
import mss
import numpy as np
from datetime import datetime, timedelta
import csv
import time
import os

bounding_box = {'top':0, 'left':960, 'width':1920, 'height':1080} 


# load yolov8 model

# load video
# cap = cv2.VideoCapture(0)

locations = [
    "harbour_bridge_east_bank",
    "houston_east_bank",
    "mon_repos",
    "ug_road",
    "vreed_en_hoop_junction",
    "harbour_bridge_west_bank",
    "sheriff_street",
    "dsl_junction",
    "diamond",
    "houston"
]

print(f"""
    Please input the number corresponding to the current location:
    0 - ${locations[0]}
    1 - ${locations[1]}
    2 - ${locations[2]}
    3 - ${locations[3]}
    4 - ${locations[4]}
    5 - ${locations[5]}
    6 - ${locations[6]}
    7 - ${locations[7]}
    8 - ${locations[8]}
    9 - ${locations[9]}
""")

# current_location = int(input("current location: "))
current_location = 0

print(f"""
    Please input the number corresponding to the destination location:
    0 - ${locations[0]}
    1 - ${locations[1]}
    2 - ${locations[2]}
    3 - ${locations[3]}
    4 - ${locations[4]}
    5 - ${locations[5]}
    6 - ${locations[6]}
    7 - ${locations[7]}
    8 - ${locations[8]}
    9 - ${locations[9]}
""")


# destination_location = int(input("destination location: "))
destination_location = 5 
# current_time = datetime.now().strftime("%H:%M:%S")
# destination_time = input("Enter the destination time: ")
# destination_time = datetime.strptime(destinati, "%H:%M:%S").strftime("%H:%M:%S")

"""
current_location
destination_location
number_of_cars_at_current_location
current_time
destination_time
average_speed_of_traffic
"""
collections = 0


data = [
]

def write_data():
    global data

    if (os.path.exists('east_bank_harbour_bridge.csv')):
        print("path is in existence")
        with open('east_bank_harbour_bridge.csv', 'a', newline='') as csvfile:
            fieldnames = ['current_location', 'destination_location', 'current_time', 'destination_time', 'number_of_cars']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerows(data)
    else:
        with open('east_bank_harbour_bridge.csv', 'w', newline='') as csvfile:
            fieldnames = ['current_location', 'destination_location', 'current_time', 'destination_time', 'number_of_cars']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)

while collections < 40:
    time.sleep(10) 
    model = YOLO('/home/shivesh/Documents/python/open_cv/computer_vision/traffic/runs/detect/train/weights/best.pt')
    current_time = datetime.now()
    destination_time = current_time + timedelta(hours=1)

    current_time = current_time.strftime("%H:%M:%S")
    destination_time = destination_time.strftime("%H:%M:%S")

    sct = mss.mss()
    ret = True
    max_count = 0
    iteration = 0
    
    while iteration < 500:
        ret = True
        frame = np.array(sct.grab(bounding_box))
        frame = frame[:, :, :3]
        
        if ret:
            results = model.track(frame, persist=True)
            
            if (results):
                ids = results[0].boxes.id
                if ids is not None:
                    count = max(ids.int().cpu().tolist())
                    print("Track ids", count)
                    max_count = count

                
            frame_ = results[0].plot()
            frame_ = cv2.resize(frame_, (int(frame_.shape[1] * 0.5), int(frame_.shape[0] * 0.5)), interpolation=cv2.INTER_AREA)  

            # visualize
            cv2.imshow('frame', frame_)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        iteration += 1
        print("Iteration", iteration)

    print("current_location", locations[current_location])
    print("destination_location", locations[destination_location])
    print("current_time", current_time)
    print("destination_time", destination_time)
    print("number_of_cars", max_count)
    
    row = {'current_location': locations[current_location], 'destination_location': locations[destination_location], 'current_time': current_time, 'destination_time': destination_time, 'number_of_cars': max_count}
    data.append(row) 
    collections+=1

write_data()
