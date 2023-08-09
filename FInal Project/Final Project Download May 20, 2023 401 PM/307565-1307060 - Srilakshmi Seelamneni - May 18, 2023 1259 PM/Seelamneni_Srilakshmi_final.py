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
FACE = 3 # Face detection Filter
SOBEL = 4 #Sobel Filter
SKIN = 5 # Skin Detector Filter
BLUR = 6 # Blur Filter

##################################################################################
# All your filter and tracker functions.
def Threshold_filter(frame):

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    h = frame.shape[0]
    w = frame.shape[1]

    imgThreshold = np.zeros((frame.shape[0],frame.shape[1]), dtype=np.uint8)

    for y in range(0,h):
    #Search each pixel in the row
        for x in range(0,w):
        
            #If the pixel value is above 100 set it to white.  If it is below 100 set it to black.
            if (frame[y,x] > 100):
                imgThreshold[y,x] = 255
            else:
                imgThreshold[y,x] = 0
    
    
    return imgThreshold
    
##################################################################################

def face_detection(frame):
    face_cascade = cv2.CascadeClassifier('C:/Users/srila/AppData/Local/Programs/Python/Python311/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')
    eyes_cascade = cv2.CascadeClassifier('C:/Users/srila/AppData/Local/Programs/Python/Python311/Lib/site-packages/cv2/data/haarcascade_eye.xml')

    # Convert to gray.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect faces in the grayscale image using the face cascade classifier.
    faces = face_cascade.detectMultiScale(gray)
    
    for (x,y,w,h) in faces:
         # Calculate the center of the face
        center = (round(x + w/2), round(y + h/2)) 
        # Draw an ellipse around the face
        frame = cv2.ellipse(frame, center, (round(w/2), round(h/2)), 0, 0, 360, (255, 0, 0), 4)  
        # Extract the region of interest corresponding to the face
        faceROI = gray[y:y+h,x:x+w]  
        # Detect eyes in the face region of interest using the eyes cascade classifier
        eyes = eyes_cascade.detectMultiScale(faceROI)
        # Iterate through all the detected eyes and draw circles around them
        for (x2,y2,w2,h2) in eyes:
            # Calculate the center of the eye
            eye_center = (x + x2 + w2//2, y + y2 + h2//2)
             # Calculate the radius of the circle  
            radius = int(round((w2 + h2)*0.25)) 
            # Draw a circle around the eye
            frame = cv2.circle(frame, eye_center, radius, (0, 0, 255 ), 4)  
        
    return frame

################################################################################################################################

def sobel_filter(frame):
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred_img = cv2.GaussianBlur(gray_image, (5,5), 5)

    sobelx = cv2.Sobel(blurred_img, cv2.CV_64F, 0, 1, ksize=3)
    sobely = cv2.Sobel(blurred_img, cv2.CV_64F, 1, 0, ksize=3)

    magnitude = cv2.magnitude(sobelx, sobely)

    threshold_value = 100
    maximum_value = 255
    threshold_type = cv2.THRESH_BINARY
    retval, thresholded = cv2.threshold(magnitude, threshold_value, maximum_value, threshold_type)

    return thresholded

########################################################################

def Skin_detection(frame):

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Minimum and maximum HSV values.
    min_HSV = np.array([0,50,100], np.uint8)
    max_HSV = np.array([20,120,250], np.uint8)
    # cv2.inRange(image, min, max)
    skinArea = cv2.inRange(img, min_HSV, max_HSV)
    # Bitwise And mask
    skinHSV = cv2.bitwise_and(img, img, mask=skinArea)
    # Convert to RGB
    skinHSV = cv2.cvtColor(skinHSV, cv2.COLOR_HSV2BGR)
    return skinHSV

################################################################################################

def Blur_filter(frame):
    blur = cv2.GaussianBlur(frame,(5,5),5)
    return blur

################################################################################################
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
        # Converting to grayscale.
        result = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    elif image_filter == FACE:
        # Converting to grayscale.
        result = face_detection(frame)
    elif image_filter == SOBEL:
        # Converting to grayscale.
        result = sobel_filter(frame)
    elif image_filter == SKIN:
        # Converting to grayscale.
        result = Skin_detection(frame)
    elif image_filter == BLUR:
        # Converting to grayscale.
        result = Blur_filter(frame)
    
    # Show your frame.  Remember you don't need to convert to RGB.  
    cv2.imshow('frame', result)
    
    # Map all your key codes to your filters/trackers. Capital and lower case.
    key = cv2.waitKey(1)

    # Quit the app.
    if key == ord('S') or key == ord('s'):
        break
    # Grayscale filter
    elif key == ord('I') or key == ord('i'):
        image_filter = GRAY
    # Threshold filter
    elif key == ord('V') or key == ord('v'):
        image_filter = THRESHOLD
    # Preview. No filter.
    elif key == ord('L') or key == ord('l'):
        image_filter = PREVIEW
    elif key == ord('U') or key == ord('u'):
        image_filter = FACE
    elif key == ord('C') or key == ord('c'):
        image_filter = SOBEL
    elif key == ord('K') or key == ord('k'):
        image_filter = SKIN
    elif key == ord('Y') or key == ord('y'):
        image_filter = BLUR

##################################################################################
# Close the app
cap.release()
cv2.destroyWindow('frame')



