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
BLUR      = 3   # Blur Filter
HSVSKIN   = 4   # HSV Skin Detector
FACEDETECT= 5   # Face Detection Filter
SOBEL     = 6   # Sobel Filter


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

def Blur_image(frame):
    blur = np.array(([.111, .111, .111],
                     [.111, .111, .111],
                     [.111,.111,.111]),
                     dtype=np.float32)
    background_subtractor = cv2.createBackgroundSubtractorMOG2()
    mask = background_subtractor.apply(frame)
    blurred = cv2.filter2D(frame, -1, blur)
    blurred[mask == 0] = frame[mask == 0]

    return blurred

def hsv_skin_detector(frame):
    ## got these values from chatGPT for a recommended mask for skin 
    lower_skin = np.array([0, 20, 70], dtype = np.uint8)
    upper_skin = np.array([20, 250, 255], dtype = np.uint8)
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    res = cv2.bitwise_and(frame, frame, mask=mask)
    return res


def face_detect(frame):
    ##I used Haar face detection for this filter
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades +"haarcascade_frontalface_default.xml")
    ##this didnt work for me
    ##cascade = cv2.CascadeClassifier('/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/cv2/data/haarcascade_frontalface_default.xml')
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5)
    for (x,y,w,h) in faces:
        center = (x + round(w/2), y + round(h/2))
        radius = int(round(w+h) * 0.25)
        cv2.circle(frame, center, radius, (255,0,0), 4)
    return frame

def sobel(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gaus_kernel = np.array([
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1]
    ], dtype = np.float32)/16
    img = cv2.filter2D(frame, -1, gaus_kernel)

    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)

    grad_mag = cv2.magnitude(sobelx, sobely)

    _, threshold  = cv2.threshold(grad_mag, 200, 255, cv2.THRESH_BINARY)
    return threshold




 
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
    elif image_filter == BLUR:
        # Send the frame to the Blur function
        result = Blur_image(frame)
    elif image_filter == HSVSKIN:
        # Send the frame to the HSV skin detector function
        result = hsv_skin_detector(frame)
    elif image_filter == FACEDETECT:
        # Send the frame to the HSV skin detector function
        result = face_detect(frame)
    elif image_filter == SOBEL:
        # Send the frame to the HSV skin detector function
        result = sobel(frame)

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
    elif key == ord('B') or key == ord('b'):
        image_filter = BLUR
    elif key == ord('H') or key == ord('h'):
        image_filter = HSVSKIN
    elif key == ord('F') or key == ord('f'):
        image_filter = FACEDETECT
    elif key == ord('S') or key == ord('s'):
        image_filter = SOBEL
                                    

##################################################################################
# Close the app
cap.release()
cv2.destroyWindow('frame')



