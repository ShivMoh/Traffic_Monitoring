import cv2
import numpy as np
import torch
from pathlib import Path
import mss
from boxmot import DeepOCSORT

model = torch.hub.load('ultralytics/yolov5', 'custom', path='/home/shivesh/Documents/python/open_cv/computer_vision/traffic/runs/train/exp20/weights/best.pt') 
screen_bounding_box = {'top': 0, 'left': 0, 'width': 1920, 'height': 1080}

tracker = DeepOCSORT(
    model_weights=Path('osnet_x0_25_msmt17.pt'), # which ReID model to use
    device='cuda:0',
    fp16=False,
)

sct = mss.mss()

# vid = cv2.VideoCapture(0)
# [144, 212, 578, 480, 0.82, 0]

init = True

while True:
    # ret, im = vid.read()
    img = np.array(sct.grab(screen_bounding_box))
    # substitute by your object detector, output has to be N X (x, y, x, y, conf, cls)
    dets = np.array([])

    results = model(img)

    image2 = np.squeeze(results.render())
    image2 = cv2.resize(image2, (int(image2.shape[1] * 0.5), int(image2.shape[0] * 0.5)), interpolation=cv2.INTER_AREA)

    # labels, cord_thres = results.xyxyn[0][:, -1].numpy(), results.xyxyn[0][:, :-1].numpy()
    # print(results.pandas().xyxy[0]) 
    # print("new line")

    if init:    
        if results.xyxy[0].shape[0] > 0:
            bounding_boxes = results.xyxy[0].cpu().numpy()

            for bounding_box in bounding_boxes:
                x1, y1, x2, y2, confidence, cls = bounding_box
                # print(x1, y1, x2, y2, confidence, cls)
                data = [x1, y1, x2, y2, confidence, cls]
                dets = np.append(dets, data)
                dets = dets.reshape(-1, len(data))
        if (dets.size > 0):
            print("Detections", dets)
            init = True
        
    # Check if there are any detections
    if dets.size > 0:
        tracker.update(dets, img) # --> M X (x, y, x, y, id, conf, cls, ind)
    # If no detections, make prediction ahead
    else:   
        print("tracking??")
        dets = np.empty((0, 6))  # empty N X (x, y, x, y, conf, cls)
        tracker.update(dets, img) # --> M X (x, y, x, y, id, conf, cls, ind)
    tracker.plot_results(img, show_trajectories=True)

    # break on pressing q or space
    img = cv2.resize(img, (int(img.shape[1] * 0.5), int(img.shape[0] * 0.5)), interpolation=cv2.INTER_AREA)  
    cv2.putText(img, str(len(dets)), (50, 50), cv2.FONT_HERSHEY_PLAIN, 10.0, (255, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow('BoxMOT tracking', img)     
    cv2.imshow('YOLOv5 detection', image2)


    key = cv2.waitKey(1) & 0xFF
    if key == ord(' ') or key == ord('q'):
        break

cv2.destroyAllWindows()