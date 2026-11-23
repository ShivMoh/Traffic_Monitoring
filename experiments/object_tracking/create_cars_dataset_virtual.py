from ultralytics import YOLO
import cv2
import mss
import numpy as np
from datetime import datetime, timedelta
import csv
import time
import os
import sys

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

current_location = int(sys.argv[1]) # 5
destination_location = int(sys.argv[2]) # 0
right = sys.argv[3]

print(current_location, destination_location, right)

collections = 0
data = []

def write_data():
    global data
    if os.path.exists('west_bank_harbour_bridge.csv'): 
        with open('west_bank_harbour_bridge.csv', 'a', newline='') as csvfile:
            fieldnames = ['current_location', 'destination_location', 'current_time', 'destination_time', 'number_of_cars']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerows(data)
    else:
        with open('west_bank_harbour_bridge.csv', 'w', newline='') as csvfile:
            fieldnames = ['current_location', 'destination_location', 'current_time', 'destination_time', 'number_of_cars']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)

cap = cv2.VideoCapture(2)

while collections < 40:
    
    model = YOLO('/home/shivesh/Documents/python/open_cv/computer_vision/traffic/runs/detect/train/weights/best.pt')
    current_time = datetime.now()
    destination_time = current_time + timedelta(hours=1)

    current_time = current_time.strftime("%H:%M:%S")
    destination_time = destination_time.strftime("%H:%M:%S")

    ret = True
    max_count = 0
    iteration = 0

    while iteration < 500:

        ret, frame = cap.read()
        if right == 'True':
            frame = frame[0:520, 0:960]
        else:
            frame = frame[0:520, 961:1920]

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

            if sys.argv[4] == '1':
                cv2.imshow(locations[current_location], frame_)
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
