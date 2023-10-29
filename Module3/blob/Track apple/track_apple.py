import cv2
import numpy as np


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error")
    exit()

while True:

    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 100, 200)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        x,y,w,h = cv2.boundingRect(contour)
        if cv2.contourArea(contour) > 25:
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0),2)

    cv2.imshow('Frame with tracked apple', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

