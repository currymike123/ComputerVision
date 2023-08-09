##################################################################################
# Libraries for the project
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt

##################################################################################
# Variables to map the numeric values to the filter names.  We will use later.
PREVIEW = 0   # Preview Mode
THRESHOLD = 1   # THRESHOLD Filter
GRAY = 2
BLUR = 3 #Convolution based filter[guassian_filter]
HSV_SKIN_DETECTOR = 4 # skin detector
SOBEL = 5 #EdgeDetection filter  
OBJ_TRACKING = 6 #tracking Algo

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

#Function to blur the given image
def blur(frame):
    #This function used to convert the given image to blur
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (3,3), 0) 
    return img_blur

#Function to convert given image to gray scale
def gray(frame):
    #returns the gray scale image
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#Function to apply sobel filter on frame
def sobel(frame):
    img_blur = blur(frame)
    # Sobel Edge Detection
    sobelx = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5) # Sobel Edge Detection on the X axis
    sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5) # Sobel Edge Detection on the Y axis
    img_mag = cv2.magnitude(sobelx, sobely)
    retval,dst = cv2.threshold(img_mag,70,255,cv2.THRESH_BINARY)
    return dst

#HSV skin detection function
def hsv_skin_detection(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Minimum and maximum HSV values.
    min_HSV = np.array([0,90,100], np.uint8)
    max_HSV = np.array([30,170,255], np.uint8)
    # # cv2.inRange(image, minimum, maximum)
    skinArea = cv2.inRange(frame, min_HSV, max_HSV)
    # # Bitwise And mask
    skinHSV = cv2.bitwise_and(frame, frame, mask=skinArea)
    # # Convert to RGB
    skinHSV = cv2.cvtColor(skinHSV, cv2.COLOR_HSV2BGR)
    return skinHSV

#Face recognition function using haarcascade from cv2 to detect face
def face_recognition(frame):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eyes_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    gray_img = gray(frame)

# Detect faces in the grayscale image using the face cascade classifier
    faces = face_cascade.detectMultiScale(gray_img)

# Iterate through all the detected faces and draw circles and ellipses around them
    for (x,y,w,h) in faces:
        center = (round(x + w/2), round(y + h/2))  # Calculate the center of the face
        radius = int(round((w + h)*0.25))  # Calculate the radius of the circle
        frame = cv2.circle(frame, center, radius, (255, 0, 0), 4)  # Draw a circle around the face
        faceROI = gray_img[y:y+h,x:x+w]  # Extract the region of interest corresponding to the face
        # Detect eyes in the face region of interest using the eyes cascade classifier
        eyes = eyes_cascade.detectMultiScale(faceROI)
        # Iterate through all the detected eyes and draw circles around them
        for (x2,y2,w2,h2) in eyes:
            eye_center = (x + x2 + w2//2, y + y2 + h2//2)  # Calculate the center of the eye
            radius = int(round((w2 + h2)*0.25))  # Calculate the radius of the circle
            frame = cv2.circle(frame, eye_center, radius, (0, 0, 255), 4)  # Draw a circle around the eye

    
    #frame=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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
        # Convert to grayscale
        result = gray(frame)
    elif image_filter == BLUR:
        result = blur(frame)
    elif image_filter == HSV_SKIN_DETECTOR:
        result = hsv_skin_detection(frame)
    elif image_filter == SOBEL:
        result = sobel(frame)
    elif image_filter == OBJ_TRACKING:
        result = face_recognition(frame)

        
    
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
    # blur filter
    elif key == ord('B') or key == ord('b'):
        image_filter = BLUR
    elif key == ord('S') or key == ord('s'):
        image_filter = SOBEL
    # Preview. No filter.
    elif key == ord('P') or key == ord('p'):
        image_filter = PREVIEW
    elif key == ord('H') or key == ord('h'):
        image_filter = HSV_SKIN_DETECTOR
    elif key == ord('F') or key == ord('f'):
        image_filter = OBJ_TRACKING

##################################################################################
# Close the app
cap.release()
cv2.destroyWindow('frame')

