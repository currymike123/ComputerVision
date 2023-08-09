import cv2
import numpy as np
import matplotlib.pyplot as plt

def Threshold_filter(frame):

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    h = frame.shape[0]
    w = frame.shape[1]

    imgThres = np.zeros((frame.shape[0],frame.shape[1]), dtype=np.uint8)

    for y in range(0,h):
    #Search each pixel in the row
        for x in range(0,w):
        
            #If the pixel value is above 100 set it to white.  If it is below 100 set it to black.
            if (frame[y,x] > 100):
                imgThres[y,x] = 255
            else:
                imgThres[y,x] = 0
    
    
    return imgThres
    
def HSV_FaceFilter(frame):
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
        return skinHSV

# Variables to map the numeric values to the filter names.
PREVIEW   = 0   # Preview Mode
HSV       = 1   # HSV FaceFilter
GRAY      = 2   # Grayscale Filter
THRESHOLD = 3   # Threshold Filter

# Create a VideoCapture object and read from input file
# If the input is taken from the camera, pass 0 instead of the video file name.
cap = cv2.VideoCapture(0)

# Variable to keep track of the current image filter.
image_filter = PREVIEW

# Video Loop
while cap.isOpened():
    # Ret = True if frame is read correctly, frame = the frame that was read.
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        break
    
    frame = cv2.resize(frame, None, fx = 0.3, fy = 0.3, interpolation=cv2.INTER_LINEAR)
    
    if image_filter == PREVIEW:
        result = frame
    elif image_filter == HSV:
        result = HSV_FaceFilter(frame)
    elif image_filter == THRESHOLD:
        result = Threshold_filter(frame)
    elif image_filter == GRAY:
        result = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    cv2.imshow('frame', result)
    
    key = cv2.waitKey(1)
    if key == ord('Q') or key == ord('q'):
        break
    elif key == ord('H') or key == ord('h'):
        image_filter = HSV
    elif key == ord('G') or key == ord('g'):
        image_filter = GRAY
    elif key == ord('T') or key == ord('t'):
        image_filter = THRESHOLD
    elif key == ord('P') or key == ord('p'):
        image_filter = PREVIEW


  
cap.release()
cv2.destroyWindow('frame')



