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
HSVSKINDE = 3   # HSV Skin detector
SHARPEN   = 4   # Sharpen Filter
GAUSBLUR  = 5   # Gausian Blur Filter
SOBEL     = 6   # Sobel Filter
FACEDETC  = 7   # Face Detection

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
def hsv_skin_detector(frame):
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    res = cv2.bitwise_and(frame, frame, mask=mask)
    return res
def sharpen(frame):
    sharpKernel = np.array([
        [-1,-1,-1],
        [-1, 9,-1],
        [-1,-1,-1]])
    sharp_frame = cv2.filter2D(frame, -1, sharpKernel)
    return sharp_frame
def gaussian_blur(frame):
    gaus_frame = cv2.filter2D(frame, -1, cv2.getGaussianKernel(3,1))
    return gaus_frame
def sobel(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    gaus_F = np.array([
        [1,2,1],
        [2,4,2],
        [1,2,1]
    ], dtype=np.float32)/16
    imgGaus = cv2.filter2D(frame, -1, gaus_F)

    sobelx = cv2.Sobel(imgGaus, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(imgGaus, cv2.CV_64F, 0, 1, ksize=5)

    # Compute the gradient magnitude 
    grad_mag = cv2.magnitude(sobelx, sobely)

    _, threshold  = cv2.threshold(grad_mag, 150, 255, cv2.THRESH_BINARY)
    return threshold
def face_detection(frame):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    # eye_cascase = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_eye.xml")
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5)
    for (x,y,w,h) in faces:
        center = (x + round(w/2), y + round(h/2))
        radius = int(round(w+h) * 0.25)
        cv2.circle(frame, center, radius, (255,0,0), 4)
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
    elif image_filter == HSVSKINDE:
        result = hsv_skin_detector(frame)
    elif image_filter == SHARPEN:
        result = sharpen(frame)
    elif image_filter == GAUSBLUR:
        result = gaussian_blur(frame)
    elif image_filter == SOBEL:
        result = sobel(frame)
    elif image_filter == FACEDETC:
        result = face_detection(frame)
    
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
    elif key == ord('S') or key == ord('s'):
        image_filter = HSVSKINDE
    elif key == ord('I') or key == ord('i'):
        image_filter = SHARPEN
    elif key == ord('B') or key == ord('b'):
        image_filter = GAUSBLUR
    elif key == ord('E') or key == ord('e'):
        image_filter = SOBEL
    elif key == ord('O') or key == ord('o'):
        image_filter = FACEDETC

##################################################################################
# Close the app
cap.release()
cv2.destroyWindow('frame')



