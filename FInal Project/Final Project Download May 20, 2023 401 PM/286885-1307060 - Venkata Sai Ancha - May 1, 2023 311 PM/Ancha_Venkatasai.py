##################################################################################
# Libraries

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the face and eyes cascade classifiers
face_cascade = cv2.CascadeClassifier('/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/cv2/data/haarcascade_frontalface_default.xml')
eyes_cascade = cv2.CascadeClassifier('/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/cv2/data/haarcascade_eye.xml')
##################################################################################
# Variables to map the numeric values to the filter names.  We will use later.
PREVIEW   = 0   # Preview Mode

# Color Space Conversion or Threshold filter using for loops.
GRAY      = 1   # Grayscale Filter
THRESHOLD = 2   # Threshold Filter
HSV = 3 #HSV Filter

# HSV Skin detector.
IMAGE = 4   # Image Pipeline Filter

# Edge-detecting pipeline with noise reduction. 
SOBEL = 5   # Sobel Pipeline Filter
CANNY = 6   # Canny Edge Detection Filter

# One object detection/tracking algorithm. 
ADABOOST = 7 # Face and Eyes detection

# Convolution-based filter. 
BLUR = 8    # Blur Filter
SHARPEN = 9 # Sharpening Filter

#######################################################################################
# All your filter and tracker functions.
def Threshold_filter(frame):
    # Convert to gray scale.
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # storing image height and width as h and w.
    h = frame.shape[0]
    w = frame.shape[1]
    # Create a blank image array to store our threshold.
    imgThres = np.zeros((frame.shape[0],frame.shape[1]), dtype=np.uint8)

    for y in range(0,h):
    # Search each pixel in the row.
        for x in range(0,w):
        
            # If the pixel value is above 100 set it to white.  If it is below 100 set it to black.
            if (frame[y,x] > 100):
                imgThres[y,x] = 255
            else:
                imgThres[y,x] = 0
    
    
    return imgThres
    
###########################################################################
def ImagePipeline_filter(frame):

    # Convert to HSV
    hsvImg = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Minimum and maximum HSV values.
    min_hsv = np.array([0,60,50], np.uint8)
    max_hsv = np.array([20,130,240], np.uint8)

    # cv2.inRange(image, minimum, maximum)
    skinArea = cv2.inRange(hsvImg, min_hsv, max_hsv)

    # Creates a structuring element "kernel" of shape cv2.MORPH_ELLIPSE with a size of 10x10 pixels
    # using the getStructuringElement function and The structuring element is used for morphological 
    # operations like erosion and dilation on an image. 
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))

    # Performs an erosion operation on the input skinArea image using the kernel structuring element with 1 iteration. 
    skinArea = cv2.erode(skinArea, kernel, iterations = 1)

    # Performs a dilation operation on the eroded skinArea image using the same kernel structuring element with 6 iterations.
    skinArea = cv2.dilate(skinArea, kernel, iterations = 6)

    # Bitwise And mask
    skinHsv = cv2.bitwise_and(frame, frame, mask = skinArea)
    
    return skinHsv

##############################################################################
def Sobel_filter(frame):
    
    # Convert to gray
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Convolve with the image
    img_blurred = cv2.GaussianBlur(img_gray, (5,5), 0)

    # Horizontal and Vertical Sobel Filtering.
    sobelx = cv2.Sobel(img_blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img_blurred, cv2.CV_64F, 0, 1, ksize=3)

    # Magnitude calculation.
    magnitude = cv2.magnitude(sobelx, sobely)

    # Thresholding.
    thresh_value = 100
    max_value = 255
    thresh_type = cv2.THRESH_BINARY
    retval, thresholded = cv2.threshold(magnitude, thresh_value, max_value, thresh_type)    

    return thresholded

###########################################################################
def canny_filter(frame):

    # Convert to gray
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # finding edges using canny edge detector.
    edge = cv2.Canny(gray,100,200)
    
    return edge

##############################################################################
def adaboost_filter(frame):

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

#################################################################################
def blur_filter(frame):
    # Blurring image by using Gaussian blur with 5x5 kernal
    blur = cv2.GaussianBlur(frame, (5,5), 0)
    return blur
    
##################################################################################
def sharpen_filter(frame):
    
    # Sharpening kernel used to sharpen the image/frame.
    kernel = np.array([[0, -1, 0],
                   [-1, 5,-1],
                   [0, -1, 0]])
    
    # using Filter2D opencv function to convolve the frame and kernel with depth 1.
    image_sharp = cv2.filter2D(src= frame, ddepth=-1, kernel=kernel)

    return image_sharp
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
    elif image_filter == HSV:
        # Convert to HSV
        result = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    elif image_filter == IMAGE:
        # Convert to IMAGE PIPELINE FILTER
        result = ImagePipeline_filter(frame)
    elif image_filter == SOBEL:
        # Convert to SOBEL filter
        result = Sobel_filter(frame)
    elif image_filter == CANNY:
        # Convert to CANNY filter
        result = canny_filter(frame)
    elif image_filter == ADABOOST:
        # Convert to ADABOOST filter
        result = adaboost_filter(frame)
    elif image_filter == BLUR:
        # Convert to BLUR filter
        result = blur_filter(frame)
    elif image_filter == SHARPEN:
        # Convert to SHARPEN filter
        result = sharpen_filter(frame)
    
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
    # HSV filter
    elif key == ord('H') or key == ord('h'):
        image_filter = HSV
    # Image pipeline filter
    elif key == ord('I') or key == ord('i'):
        image_filter = IMAGE
    # Sobel filter
    elif key == ord('S') or key == ord('s'):
        image_filter = SOBEL
    # Canny edge detector
    elif key == ord('C') or key == ord('c'):
        image_filter = CANNY
    # Adaboost filter
    elif key == ord('A') or key == ord('a'):
        image_filter = ADABOOST
    # Blur filter
    elif key == ord('B') or key == ord('b'):
        image_filter = BLUR
    # Sharpen filter
    elif key == ord('N') or key == ord('n'):
        image_filter = SHARPEN
##################################################################################
# Close the app
cap.release()
cv2.destroyWindow('frame')



