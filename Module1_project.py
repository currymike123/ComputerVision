import cv2
import numpy as np
import matplotlib.pyplot as plt

# Create a VideoCapture object and read from input file
# If the input is taken from the camera, pass 0 instead of the video file name.
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        break
    
    # frame =cv2.rotate(frame, cv2.ROTATE_180)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # # Minimum and maximum HSV values.
    min_HSV = np.array([0,90,100], np.uint8)
    max_HSV = np.array([30,170,255], np.uint8)

    # # cv2.inRange(image, minimum, maximum)
    skinArea = cv2.inRange(frame, min_HSV, max_HSV)

    # # Bitwise And mask
    skinHSV = cv2.bitwise_and(frame, frame, mask=skinArea)

    # # Convert to RGB
    skinHSV = cv2.cvtColor(skinHSV, cv2.COLOR_HSV2BGR)

    cv2.imshow('frame', skinHSV)
    if cv2.waitKey(1) == ord('q'):
        break

  
cap.release()
cv2.destroyWindow('frame')