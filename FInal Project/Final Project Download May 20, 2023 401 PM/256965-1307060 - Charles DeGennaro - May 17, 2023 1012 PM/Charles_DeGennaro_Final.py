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
SKIN      = 3   # HSV Skin Detector
BLUR      = 4   # Gaussian Blur
EDGE      = 5   # Sobel Edge Detector
OBJECT    = 6   # Object Detection

##################################################################################
# All your filter and tracker functions.
def threshold_filter(frame):

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    h = frame.shape[0]
    w = frame.shape[1]

    img_thres = np.zeros((frame.shape[0],frame.shape[1]), dtype=np.uint8)

    for y in range(0,h):
    #Search each pixel in the row
        for x in range(0,w):
        
            #If the pixel value is above 100 set it to white.  If it is below 100 set it to black.
            if (frame[y,x] > 100):
                img_thres[y,x] = 255
            else:
                img_thres[y,x] = 0
    
    
    return img_thres

def gaussian_blur(frame):
    gaussian_filter = np.array([[1,2,1],[2,4,2],[1,2,1]], dtype=np.float32) / 16
    
    for _ in range(1):
        frame = cv2.filter2D(frame, -1, gaussian_filter)
    return frame

def sobel(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    blur = gaussian_blur(gray)
    
    # cv2.imshow('frame', blur)
    
    # Sobel kernels
    kernelX = np.array([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=np.float32)
    kernelY = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=np.float32)
    # Sobel convolution
    imgSobelX = cv2.filter2D(blur, cv2.CV_64F, kernelX)
    imgSobelY = cv2.filter2D(blur, cv2.CV_64F, kernelY)
    
    sobel = (cv2.Sobel(blur, cv2.CV_64F, 1, 1, 5) * 20).clip(0,255).astype(np.uint8)
    ret, sobel = cv2.threshold(sobel, 200, 255, cv2.THRESH_BINARY)
    
    # sobel = cv2.magnitude(imgSobelX, imgSobelY)
    
    # ret = np.array(sobel).astype(int)
    
    # print(ret)
    
    return sobel

def skin(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Create min and max HSV values for mask
    min_HSV = np.array([0,20,50], np.uint8)
    max_HSV = np.array([30,150,255], np.uint8)
    
    # cv2.inRange(image, min, max)
    skin_area = cv2.inRange(frame, min_HSV, max_HSV)
    
    # Bitwise AND mask
    skin_HSV = cv2.bitwise_and(frame, frame, mask=skin_area)
    
    bgr = cv2.cvtColor(skin_HSV, cv2.COLOR_HSV2BGR)
    
    return bgr

def object(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    edges = cv2.Canny(gray, 100, 200)
    
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    min_contour_area = 500
    contours_large = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_contour_area:
            contours_large.append(cnt)

    for cnt in contours_large:
        x,y,w,h = cv2.boundingRect(cnt)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),1)
    
    return frame
    
##################################################################################
# The video image loop.

# Create a VideoCapture object and read from input file
# If the input is taken from the camera, pass 0 instead of the video file name.
cap = cv2.VideoCapture(0)

# Variable to keep track of the current image filter. Default set to Preview.
image_filter = PREVIEW

# Video Loop
while cap.isOpened():
    # Ret = True if frame is read correctly, frame = the frame that was read.
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        break
    
    # Resize the image so your app runs faster.  Less Pixels to Process.  
    # Use Nearest Neighbor Interpolation for the fastest runtime.
    frame = cv2.resize(frame, None, fx = 0.4, fy = 0.4, interpolation=cv2.INTER_NEAREST)
    
    # Send your frame to each filter.
    if image_filter == PREVIEW:
        # No filter.
        result = frame
    elif image_filter == THRESHOLD:
        # Send the frame to the Threshold function.
        result = threshold_filter(frame)
    elif image_filter == GRAY:
        # Convert to grayscale.
        result = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    elif image_filter == BLUR:
        # Convert to grayscale.
        result = gaussian_blur(frame)
    elif image_filter == EDGE:
        # Apply Sobel Edge Detector.
        result = sobel(frame)
    elif image_filter == SKIN:
        # Apply Skin Detector.
        result = skin(frame)
    elif image_filter == OBJECT:
        # Apply Object Detector.
        result = object(frame)
    
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
    # Gaussian Blur
    elif key == ord('B') or key == ord('b'):
        image_filter = BLUR
    # Sobel Edge Detector
    elif key == ord('E') or key == ord('e'):
        image_filter = EDGE
    # Skin detector
    elif key == ord('S') or key == ord('s'):
        image_filter = SKIN
    # Object detector
    elif key == ord('O') or key == ord('o'):
        image_filter = OBJECT

##################################################################################
# Close the app
cap.release()
cv2.destroyWindow('frame')



