import cv2 as cv
import numpy as np

image = cv.imread("./source_image.png")

while True:
    
    cv.imshow("Image", image)
    
    ret1, corners1 = cv.findChessboardCorners(image, (800, 800))
    
    print(ret1, corners1)
    # cv.imshow("Image 2", corners1)

    if cv.waitKey(0) & 0xFF==ord('d'):
        break

