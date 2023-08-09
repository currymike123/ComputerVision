# import libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Import video file
cap = cv2.VideoCapture('Graphics/coco_moving.mp4')

# Has the file opened correctly.
while cap.isOpened():

    # Is there a current frame.  Frame pixel array.
    ret, frame = cap.read()

    # If there are no more frames break the loop.
    if not ret:
        break

    cv2.imshow(frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyWindow('frame')

