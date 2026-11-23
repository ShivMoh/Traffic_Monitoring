import cv2 as cv
from ultralytics import YOLO, solutions
import numpy as np

model = YOLO('/home/shivesh/Documents/python/open_cv/computer_vision/traffic/runs/detect/train/weights/best.pt')
names = model.model.names

cap = cv.VideoCapture("./sample.mp4")
points = [(330, 380), (1030, 380)]

# points  = [(int(x), int(y)) for x, y in points]

w, h, fps = (int(cap.get(x)) for x in (cv.CAP_PROP_FRAME_WIDTH, cv.CAP_PROP_FRAME_HEIGHT, cv.CAP_PROP_FPS))
video_writer = cv.VideoWriter("speed_estimation.avi", cv.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

speed_obj = solutions.SpeedEstimator(
    reg_pts=points,
    names=names,
    view_img=True
)

print(speed_obj.reg_pts)
while cap.isOpened():
    
    ret, frame = cap.read()
     
    if not ret:
        break
    
    tracks = model.track(frame, persist=True, show=False)
    
    annotated_image = speed_obj.estimate_speed(frame, tracks)
    
    video_writer.write(annotated_image)
    
    cv.imshow("Annotated frame", frame)
    # cv.imshow("Frame", frame)
    # cv.imshow("Speed Frame", np.array(speed_frame))

    if cv.waitKey(25) & 0xFF == ord('q'):
       break 

cap.release()
cv.destroyAllWindows()
cv.destroyAllWindows()
