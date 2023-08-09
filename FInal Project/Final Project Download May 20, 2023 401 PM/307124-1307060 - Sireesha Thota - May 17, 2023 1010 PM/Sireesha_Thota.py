##################################################################################
# Libraries

import cv2
import numpy as np
import matplotlib.pyplot as plt

##################################################################################g
# Variables to map the numeric values to the filter names.  We will use later.
PREVIEW   = 0   # Preview Mode
GRAY      = 1   # Grayscale Filter
THRESHOLD = 2   # Threshold Filter
SKIN      = 3   #HSV-SKin Filter
SHARP     = 4   #Sharppening Filter
BLUR      = 5   #Blurring Filter
EDGE      = 6   #EDGE_SOBEL Filter
FACE      = 7   #FACE Filter
##################################################################################
#Sharpen and blur arrays for convolution
sharpen = np.array(([0,-1,0],
                    [-1,5,-1],
                    [0,-1,0]),
                    dtype=np.float32)

blur = np.array(([.111,.111,.111],
                   [.111,.111,.111],
                   [.111,.111,.111]),
                   dtype=np.float32)

# All your filter and tracker functions.

#---------------------------------------------------------------
#Threshold_filter
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
#---------------------------------------------------------------
#HSV_Skin Filter
def HSV_SKIN(frame):

    #frame =cv2.rotate(frame, cv2.ROTATE_180)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # # Minimum and maximum HSV values.
    min_HSV = np.array([0,70,100], np.uint8)
    max_HSV = np.array([25,100,250], np.uint8)

    # # cv2.inRange(image, minimum, maximum)q
    skinArea = cv2.inRange(frame, min_HSV, max_HSV)

    # # Bitwise And mask
    skinHSV = cv2.bitwise_and(frame, frame, mask=skinArea)

    # # Convert to RGB
    skinHSV = cv2.cvtColor(skinHSV, cv2.COLOR_HSV2BGR)
    return skinHSV
#---------------------------------------------------------------
#Sobel_Edge_Filter
def sobel_edge_filter(frame):
    frame=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #Adding blur filter
    img_blur = cv2.GaussianBlur(frame, (5,5), 0)

    # Applying Sobel operators to calculate horizontal and vertical gradients
    Sobel_Gradx = cv2.Sobel(img_blur, cv2.CV_64F, 1, 0)
    Sobel_Grady = cv2.Sobel(img_blur, cv2.CV_64F, 0, 1)

    # Calculating the magnitude of gradient using cv2.magnitude
    magnitude = cv2.magnitude(Sobel_Gradx, Sobel_Grady)

    # Applying the threshold to the magnitude image to get a binary edge map
    ret, edge_map = cv2.threshold(magnitude, 50, 255, cv2.THRESH_BINARY)
    return edge_map
#---------------------------------------------------------------
# Face Recognition filter
def face_filter(frame):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eyes_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Detect faces in the grayscale image using the face cascade classifier
    faces = face_cascade.detectMultiScale(gray)

# Iterate through all the detected faces and draw circles and ellipses around them
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
        # Convert to grayscale.
        result = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    elif image_filter == SKIN:
        # Send the frame to the HSV_SKIN function.
        result = HSV_SKIN(frame)
    elif image_filter == SHARP:
        # Add Sharpen filter
        result = cv2.filter2D(frame,-1, sharpen)
    elif image_filter == BLUR:
        # Add Blurring filter.
        result = cv2.filter2D(frame,-1, blur)
    elif image_filter == EDGE:
        # Send the frame to the sobel edge filter function.
        result = sobel_edge_filter(frame)
    elif image_filter == FACE:
        # Send the frame to the face recognition filter function.
        result = face_filter(frame)

   
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
    # HSV_Skin Filter
    elif key == ord('H') or key == ord('h'):
        image_filter = SKIN
    # Sharpen filter
    elif key == ord('S') or key == ord('s'):
        image_filter = SHARP
    # Blur filter
    elif key == ord('B') or key == ord('b'):
        image_filter = BLUR
    # Sobel Edge Filter
    elif key == ord('E') or key == ord('e'):
        image_filter = EDGE
    # Face Recognition filter
    elif key == ord('F') or key == ord('f'):
        image_filter = FACE

##################################################################################
# Close the app
cap.release()
cv2.destroyWindow('frame')
