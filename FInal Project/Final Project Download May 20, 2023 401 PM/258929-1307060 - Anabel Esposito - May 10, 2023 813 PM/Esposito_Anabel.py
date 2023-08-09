##################################################################################
# Libraries

import cv2
import numpy as np
import matplotlib.pyplot as plt
import site
print(site.getsitepackages())

# Load the face and eyes cascade classifiers
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eyes_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')


##################################################################################
# Variables to map the numeric values to the filter names.  We will use later.
PREVIEW   = 0   # Preview Mode
GRAY      = 1   # Grayscale Filter
THRESHOLD = 2   # Threshold Filter
GAUSSIAN  = 3   # Gaussian Blur Filter
SOBEL     = 4   # Sobel Filter
HSVSKIN   = 5   # HSV Skin Detection Filter
OBJDETECT = 6   # Object Detection Filter

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
    elif image_filter == GAUSSIAN:
        # Convert to Gaussian Blur.
        gaussian_filter = np.array([[1,2,1],[2,4,2],[1,2,1]], dtype=np.float32) / 16
        result = cv2.filter2D(frame, -1, gaussian_filter)
    elif image_filter == SOBEL:
        # Convert to Sobel
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gaussian_filter = np.array([[1,2,1],[2,4,2],[1,2,1]], dtype=np.float32) / 16
        imgGaussian = cv2.filter2D(frame, -1, gaussian_filter)
        sobelx = cv2.Sobel(imgGaussian, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(imgGaussian, cv2.CV_64F, 0, 1, ksize=3)
        mag = cv2.magnitude(sobelx, sobely)
        th, imgOcvThres = cv2.threshold(mag, 100, 255, cv2.THRESH_BINARY)
        result = imgOcvThres
    elif image_filter == HSVSKIN:
        # Convert to HSV Skin detecting.
        min_HSV = np.array([0,45,0], np.uint8)
        max_HSV = np.array([135,135,250], np.uint8)

        # cv2.inRange(image, minimum, maximum)
        skinArea = cv2.inRange(frame, min_HSV, max_HSV)
        # Bitwise And mask
        skinHSV = cv2.bitwise_and(frame, frame, mask=skinArea)
        result = skinHSV
    elif image_filter == OBJDETECT:
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray)

        for (x,y,w,h) in faces:
            center = (round(x + w/2), round(y + h/2))  # Calculate the center of the face
            radius = int(round((w + h)*0.25))  # Calculate the radius of the circle
            frame = cv2.circle(frame, center, radius, (255, 0, 0), 4)  # Draw a circle around the face
            faceROI = gray[y:y+h,x:x+w]  # Extract the region of interest corresponding to the face
            # Detect eyes in the face region of interest using the eyes cascade classifier
            eyes = eyes_cascade.detectMultiScale(faceROI)
            # Iterate through all the detected eyes and draw circles around them
            for (x2,y2,w2,h2) in eyes:
                eye_center = (x + x2 + w2//2, y + y2 + h2//2)  # Calculate the center of the eye
                radius = int(round((w2 + h2)*0.25))  # Calculate the radius of the circle
                frame = cv2.circle(frame, eye_center, radius, (0, 0, 255), 4)  # Draw a circle around the eye
        result = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    
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
    # Gaussian Blur filter
    elif key == ord('B') or key == ord('b'):
        image_filter = GAUSSIAN
    # HSV skin filter
    elif key == ord('H') or key == ord('h'):
        image_filter = HSVSKIN
    # SOBEL filter
    elif key == ord('S') or key == ord('s'):
        image_filter = SOBEL
    # Object Detection filter
    elif key == ord('O') or key == ord('o'):
        image_filter = OBJDETECT
    # Preview. No filter.
    elif key == ord('P') or key == ord('p'):
        image_filter = PREVIEW

##################################################################################
# Close the app
cap.release()
cv2.destroyWindow('frame')



