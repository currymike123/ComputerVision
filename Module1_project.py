import cv2
import numpy as np
import matplotlib.pyplot as plt

# Create a VideoCapture object and read from input file
# If the input is taken from the camera, pass 0 instead of the video file name.
cap = cv2.VideoCapture(0)


# Video Loop
while cap.isOpened():
    # Ret = True if frame is read correctly, frame = the frame that was read.
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        break
    
     # Convert to HSV
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Minimum and maximum HSV values.
    min_HSV = np.array([0,50,100], np.uint8)
    max_HSV = np.array([40,170,255], np.uint8)

    # cv2.inRange(image, minimum, maximum)
    skinMask = cv2.inRange(frame, min_HSV, max_HSV)

    kernel = np.ones((3,3), dtype='uint8')
    # skinMask = cv2.erode(skinMask, kernel, iterations = 5)
    skinMask = cv2.dilate(skinMask, kernel, iterations = 2)

    # Bitwise And mask
    skinHSV = cv2.bitwise_and(frame, frame, mask=skinMask)

    # Convert to RGB
    skinHSV = cv2.cvtColor(skinHSV, cv2.COLOR_HSV2BGR)
    
    cv2.imshow('frame', skinHSV)
    
    key = cv2.waitKey(1)
    if key == ord('Q') or key == ord('q'):
        break
   
  
cap.release()
cv2.destroyWindow('frame')



