##################################################################################
# Libraries

import cv2
import numpy as np
import matplotlib.pyplot as plt

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eyes_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

##################################################################################
# Variables to map the numeric values to the filter names.  We will use later.
PREVIEW   = 0   # Preview Mode
GRAY      = 1   # Grayscale Filter
THRESHOLD = 2   # Threshold Filter
SHARPEN = 3     # Sharpening Filter
BLUR = 4        # Gaussian Blurring Filter
SOBEL = 5       # Sobel Filter
HSV = 6         # HSV Skin Detection
OBJDETECT = 7   # Object Detection

##################################################################################
# All your filter and tracker functions.
# Code for 3x3 kernel with convolution. No Padding. Stride = 1.
# To perform convolution on the whole image padding needs to be added.

# Threshold Filter
def Threshold_Filter(frame):

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    h = frame.shape[0]
    w = frame.shape[1]

    image_thresh = np.zeros((frame.shape[0],frame.shape[1]), dtype=np.uint8)

    for y in range(0,h):
    #Search each pixel in the row
        for x in range(0,w):
        
            #If the pixel value is above 100, set it to white.
            if (frame[y,x] > 100):
                image_thresh[y,x] = 255
            else:
            #If it is below 100, set it to black.
                image_thresh[y,x] = 0
    
    return image_thresh

# Sharpening Filter
def Sharpening_Filter(frame):
    #sharpening kernel
    kernel = np.array(([0,-1,0],[-1,5,-1],[0,-1,0]), dtype=np.float32)
    
    image_sharpened = cv2.filter2D(frame, -1, kernel)
    return image_sharpened

# Gaussian Blurring Filter
def Blurring_Filter(frame):
    #gaussian blur kernel:
    gaussian_filter = np.array([[1,2,1],[2,4,2],[1,2,1]], dtype=np.float32) / 16

    #add filter to img
    image_blurred = cv2.filter2D(frame, -1, gaussian_filter)
    return image_blurred

# Sobel Filter
def Sobel_Filter(frame):
    #grayscale frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #gaussian blur
    image_blurred = Blurring_Filter(gray)

    #vertical + horizontal sobel filters
    sobelx = cv2.Sobel(image_blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(image_blurred, cv2.CV_64F, 0, 1, ksize=3)

    #combine sobelx and sobely
    magnitude = cv2.magnitude(sobelx, sobely)
    
    #threshold filter
    th, image_sobel = cv2.threshold(magnitude, 100, 255, cv2.THRESH_BINARY);
    return image_sobel
    
# HSV Skin Detector:
def HSV_Skin_Detector(frame):
    min = np.array([0,50,0], np.uint8)
    max = np.array([130,130,225], np.uint8)

    skinArea = cv2.inRange(frame, min, max)
    image_HSV = cv2.bitwise_and(frame, frame, mask=skinArea)

    return image_HSV

# Object Detection
def Object_Detection(frame): 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #detect faces in img using the face cascade classifier
    faces = face_cascade.detectMultiScale(gray)

    #iterate through all faces and draw ellipses around them
    for (x,y,w,h) in faces:
        center = (round(x + w/2), round(y + h/2)) #center of face
        frame = cv2.ellipse(frame, center, (round(w/2), round(h/2)), 0, 0, 360, (255, 0, 0), 4)
        faceROI = gray[y:y+h,x:x+w] #region of interest
        #detect eyes in face using eyes cascade classifier
        eyes = eyes_cascade.detectMultiScale(faceROI)
 
        #iterate through all eyes and draw circles around them
        for (x2,y2,w2,h2) in eyes:
            eye_center = (x + x2 + w2//2, y + y2 + h2//2)  
            radius = int(round((w2 + h2)*0.25)) 
            frame = cv2.circle(frame, eye_center, radius, (0, 0, 255), 4)  

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
    frame = cv2.resize(frame, None, fx = 1, fy = 1, interpolation=cv2.INTER_NEAREST)
    
    # Send your frame to each filter.
    if image_filter == PREVIEW:
        # No filter.
        result = frame
    elif image_filter == BLUR:
        # Send the frame to the Blurring function.
        result = Blurring_Filter(frame)
    elif image_filter == SHARPEN:
        # Send the frame to the Sharpening function.
        result = Sharpening_Filter(frame)
    elif image_filter == THRESHOLD:
        # Send the frame to the Threshold function.
        result = Threshold_Filter(frame)
    elif image_filter == SOBEL:
        # Send the frame to the Sobel function.
        result = Sobel_Filter(frame)
    elif image_filter == HSV:
        # Send the frame to the HSV Skin Detection function.
        result = HSV_Skin_Detector(frame)
    elif image_filter == OBJDETECT:
        # Send the frame to the Object Detection function.
        result = Object_Detection(frame)
    elif image_filter == GRAY:
        # Convert to grayscale.
        result = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
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
    # Blurring filter
    elif key == ord('B') or key == ord('b'):
        image_filter = BLUR
    # Sharpening filter
    elif key == ord('K') or key == ord('k'):
        image_filter = SHARPEN
    # Threshold filter
    elif key == ord('T') or key == ord('t'):
        image_filter = THRESHOLD
    # Sobel filter
    elif key == ord('S') or key == ord('s'):
        image_filter = SOBEL
    # HSV Skin Detection
    elif key == ord('H') or key == ord('h'):
        image_filter = HSV
    # Object Detection
    elif key == ord('O') or key == ord('o'):
        image_filter = OBJDETECT
    # Preview. No filter.
    elif key == ord('P') or key == ord('p'):
        image_filter = PREVIEW
##################################################################################
# Close the app
cap.release()
cv2.destroyWindow('frame')