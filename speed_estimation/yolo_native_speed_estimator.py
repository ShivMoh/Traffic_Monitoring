import cv2 as cv
from ultralytics import YOLO, solutions
import mss
import numpy as np
import math

sct = mss.mss()

bounding_box = {'top':0, 'left':0, 'width':1920, 'height':1080} 
model = YOLO('/home/shivesh/Documents/python/open_cv/computer_vision/traffic/runs/detect/train/weights/best.pt')
names = model.model.names

# boundary_1 = [(330, 380), (330, 450)]
# boundary_2 = [(1030, 380), (1030, 450)]
points = [(int(1920/2), 0), (int(1920/2), 1080)]
w = 1920
h = 1080
fps = 1/25
speed_region = [(0, 0), (1920, 0), (1920, 1080), (0, 1080)]

speed_obj = solutions.SpeedEstimator(
    reg_pts=points,
    names=names,
    view_img=True,
)

current_count = 0

while True:
    raw_frame = np.array(sct.grab(bounding_box))
    
    frame = raw_frame[:, :, :3]
    result = model.track(frame, persist=True, show=False)
     
    if result:
        speed_frame = speed_obj.estimate_speed(raw_frame, result)
        if speed_frame is not None:
            frame = speed_frame

    cv.imshow("Frame", frame)

    if cv.waitKey(25) & 0xFF == ord('q'):
       break 

cv.destroyAllWindows()
