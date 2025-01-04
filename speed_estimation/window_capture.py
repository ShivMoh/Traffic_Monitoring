import cv2 as cv

cap = cv.VideoCapture(2)

if not cap.isOpened():
    quit()

while True:

    ret, frame = cap.read()

    if not ret:
        print("no frame")
        continue

    cv.imshow("frame", frame)

    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()

