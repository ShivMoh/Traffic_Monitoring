import time
import torch
import cv2
import mss
import numpy as np

# Load model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='./yolov5/runs/train/exp20/weights/best.pt') 
bounding_box = {'top': 0, 'left': 0, 'width': 1920, 'height': 1080}

sct = mss.mss()

while True:
    last_time = time.time()

    # Get raw pixels from the screen, save it to a Numpy array
    img = np.array(sct.grab(bounding_box))

    results = model(img)

    
    # Display the picture
    cv2.imshow('test', np.squeeze(results.render()))
    print(results)

    # Display the picture in grayscale
    # cv2.imshow('OpenCV/Numpy grayscale',
    #            cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY))

    # print("fps: {}".format(1 / (time.time() - last_time)))

    if cv2.waitKey(25) & 0xFF == 27:
        cv2.destroyAllWindows()
        break