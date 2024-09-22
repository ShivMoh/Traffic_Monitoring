import supervision as sv
import cv2 as cv
from ultralytics import YOLO
from mss import mss
import numpy as np

model = YOLO('/home/shivesh/Documents/python/open_cv/computer_vision/traffic/runs/detect/train/weights/best.pt')

bounding_box = {'top':0, 'left':0, 'width':1920, 'height':1080} 
bounding_box_annotator = sv.BoxAnnotator()

sct = mss()

# cap = cv.VideoCapture(0)

byte_track = sv.ByteTrack()

while True:
    print("hello")
    frame = np.array(sct.grab(bounding_box))
    frame = frame[:,:, :3] 

    result = model(frame)[0]
    detections = sv.Detections.from_ultralytics(result)

    annotated_frame = bounding_box_annotator.annotate(
            scene=frame.copy(),
            detections=detections
   )

    cv.imshow("showing screen", annotated_frame)

    if cv.waitKey(25) & 0xFF==ord('q'):
        break

print("this should be running one")
