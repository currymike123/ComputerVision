##################################################################################
# Libraries

import cv2
import numpy as np
import matplotlib.pyplot as plt

##################################################################################
# Variables to map the numeric values to the filter names.  We will use later.
PREVIEW   = 0   # Preview Mode
GRAY      = 1   # Grayscale Filter
THRESHOLD = 2   # Threshold Filter
SHARPEN   = 3   # Sharpening Filter
HSV       = 4   # HSV Filter
EDGE      = 5   # EdgeDetection Filter
OBJECT    = 6   # ObjectDetection Filter


##################################################################################
# All your filter and tracker functions.
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

def EdgeDetection_Filter(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(frame, (3,3),0)

    sobelX = cv2.Sobel(blur,cv2.CV_64F, 1,0,ksize=3)
    sobelY = cv2.Sobel(blur,cv2.CV_64F, 0,1, ksize=3)

    mag = cv2.magnitude(sobelX,sobelY)

    val,thresh = cv2.threshold(mag, 120, 255, cv2.THRESH_BINARY)

    return thresh

def HSV_Conversion(frame):
    frameHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    return frameHSV

def convolve(frame,kernel):
    frameConv2 = cv2.filter2D(frame,-1, kernel)
    return frameConv2

def ObjectDetection(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection
    edges = cv2.Canny(gray, 100, 200)

    # Find contours and hierarchy
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Loop through the hierarchy array
    for i in range(len(contours)):
        # Filter contours based on their hierarchy
        if hierarchy[0][i][3] == -1:
            # This is an outermost contour
            cv2.drawContours(frame, contours, i, (0, 255, 0), 2)
        elif hierarchy[0][i][3] == 0:
            # This is a contour with no parent
            cv2.drawContours(frame, contours, i, (255, 0, 0), 2)
        else:
            # This is a child contour
            cv2.drawContours(frame, contours, i, (0, 0, 255), 2)

            return frame

    

##################################################################################
# The video image loop.

# Create a VideoCapture object and read from input file
# If the input is taken from the camera, pass 0 instead of the video file name.
cap = cv2.VideoCapture(0)

# Variable to keep track of the current image filter. Default set to Preview.
image_filter = PREVIEW

# Kernel for convolution
sharpen = np.array(([0,-1,0],
[-1,5,-1],
[0,-1,0]),
dtype=np.float32)



# Video Loop
while cap.isOpened():
    # Ret = True if frame is read correctly, frame = the frame that was read.
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        break
    
    # Resize the image so your app runs faster.  Less Pixels to Process.  
    # Use Nearest Neighbor Interpolation for the fastest runtime.
    frame = cv2.resize(frame, None, fx = 0.3, fy = 0.3, interpolation=cv2.INTER_NEAREST)
    
    # Send your frame to each filter.
    if image_filter == PREVIEW:
        # No filter.
        result = frame
    elif image_filter == THRESHOLD:
        # Send the frame to the Threshold function.
        result = Threshold_filter(frame)
    elif image_filter == GRAY:
        # Convert to grayscale.
        result = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    elif image_filter == SHARPEN:
        result = convolve(frame,sharpen)
    elif image_filter == HSV:
        result = HSV_Conversion(frame)
    elif image_filter == EDGE:
        result = EdgeDetection_Filter(frame)
    elif image_filter == OBJECT:
        result = ObjectDetection(frame)
    
    # Show your frame.  Remember you don't need to convert to RGB.  
    cv2.imshow('frame', result)
    
    # Map all your key codes to your filters/trackers. Capital and lower case.
    key = cv2.waitKey(1)

    # Quit the app.
    if key == ord('Q') or key == ord('q'):
        break
    # Grayscale filter
    elif key == ord('G') or key == ord('g'):
        image_filter = GRAY
    # Threshold filter
    elif key == ord('T') or key == ord('t'):
        image_filter = THRESHOLD
    # Preview. No filter.
    elif key == ord('P') or key == ord('p'):
        image_filter = PREVIEW
    # Sharpening Filter
    elif key == ord('S') or key == ord('s'):
        image_filter = SHARPEN
    # HSV Filter
    elif key == ord('H') or key == ord('h'):
        image_filter = HSV
    # Sobel Filter
    elif key == ord('E') or key == ord('e'):
        image_filter = EDGE
    # Object Detecting Filter
    elif key == ord('O') or key == ord('o'):
        image_filter = OBJECT


##################################################################################
# Close the app
cap.release()
cv2.destroyWindow('frame')



