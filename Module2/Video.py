### Video Loop

import cv2
import numpy as np

####################################################################

# Filter Variables to numeric values.
PREVIEW = 0
GRAY = 1
THRESHOLD = 2

####################################################################

def Threshold(frame):

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    h = frame.shape[0]
    w = frame.shape[1]

    imgThres = np.zeros((h,w), dtype=np.uint8)

    for y in range(0,h):
        for x in range(0,w):

            if(frame[y,x] > 100):
                imgThres[y,x] = 255
            else:
                imgThres[y,x] = 0

    return imgThres


####################################################################

# Create a Video Capture Object and read from input file or the webcam.  
# If the input is taken from a camera or the webcam, pass 0.
# If it's a file path and file name.

cap = cv2.VideoCapture(0)

# Variable to keep track of our current image filter. Default to Preview.
image_filter = PREVIEW

# Video loop
while cap.isOpened():

    # Ret = True if frame is read correctly. frame = the current video frame.
    ret, frame = cap.read()

    # If frame is read correctly ret is True
    if not ret:
        break 

    # Resize the frame with Nearest Neighbor Interpolation.
    frame = cv2.resize(frame, None, fx = 0.3, fy = 0.3, interpolation=cv2.INTER_NEAREST)
    
    if image_filter == PREVIEW:
        result = frame
    elif image_filter == THRESHOLD:
        result = Threshold(frame)
    elif image_filter == GRAY:
        result = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Show current frame.
    cv2.imshow('frame', result)

    # Map keyboard input to filters/trackers. Capital and lower case.
    key = cv2.waitKey(1)

    # Quit the app.
    if key == ord('Q') or key == ord('q'):
        break
    if key == ord('T') or key == ord('t'):
        image_filter = THRESHOLD
    if key == ord('P') or key == ord('p'):
        image_filter = PREVIEW
    if key == ord('G') or key == ord('g'):
        image_filter = GRAY

#Close the app
cap.release()
cv2.destroyWindow('frame')

