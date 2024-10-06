import cv2 as cv
from ultralytics import YOLO, solutions
import mss
import numpy as np

sct = mss.mss()

bounding_box = {'top':0, 'left':0, 'width':1920, 'height':1080} 
model = YOLO('/home/shivesh/Documents/python/open_cv/computer_vision/traffic/runs/detect/train/weights/best.pt')
names = model.model.names

# boundary_1 = [(330, 380), (330, 450)]
# boundary_2 = [(1030, 380), (1030, 450)]
points = [(0, 380), (1920, 380)]
w = 1920
h = 1080
fps = 1/25

speed_obj = solutions.SpeedEstimator(
    reg_pts=points,
    names=names,
    view_img=True
)

while True:
    raw_frame = np.array(sct.grab(bounding_box))
    
    frame = raw_frame[:, :, :3]
    tracks = model.track(frame, persist=True, show=False)
    
    if tracks:
        speed_frame = speed_obj.estimate_speed(raw_frame, tracks)
        if speed_frame is not None:
            print("showing frame")
            cv.imshow("Frame", speed_frame)

    # cv.imshow("Speed Frame", np.array(speed_frame))


    if cv.waitKey(25) & 0xFF == ord('q'):
       break 

cv.destroyAllWindows()
