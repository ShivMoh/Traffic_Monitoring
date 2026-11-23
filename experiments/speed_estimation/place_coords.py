from types import FrameType
import numpy as np
import cv2 as cv
 
drawing = False # true if mouse is pressed
mode = True # if True, draw rectangle. Press 'm' to toggle to curve
ix,iy = -1,-1
 
# mouse callback function
def draw_circle(event,x,y,flags,param):
    global img, points  
    if event == cv.EVENT_LBUTTONDOWN:
        print(event, x, y)
        points.append((x,y))
        cv.circle(img, (x,y), 10, (255, 0, 0), -1)
 
 
img = np.zeros((512,512,3), np.uint8)
cv.namedWindow('image')
cv.setMouseCallback('image',draw_circle)
 
while(1):
    cv.imshow('image',img)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
 
cv.destroyAllWindows()
