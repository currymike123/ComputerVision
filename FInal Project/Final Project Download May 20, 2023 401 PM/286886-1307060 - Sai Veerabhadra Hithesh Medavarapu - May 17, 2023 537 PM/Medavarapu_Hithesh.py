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
SKIN_DETECTION = 3  # Skin Detection (In-Class Project)
EDGE_DETECTION = 4 # Edge Detection (Sobel filter)
BLUR_FILTER = 5 # Blurring using kernel 
SHARPEN_FILTER = 6 # Sharpening using kernel
FACE_EYES_FILTER = 7 # Face Detection (Haar-cascade)
CONTOUR_TRACK = 8 # Countours Tracking 

##################################################################################
# All your filter and tracker functions.

#THRESHOLD FILTER USING FOR LOOPS (press T)

def Threshold_filter(frame):

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    h = frame.shape[0]
    w = frame.shape[1]

    imgThres = np.zeros((frame.shape[0],frame.shape[1]), dtype=np.uint8)

    for y in range(0,h):
    #Search each pixel in the row
        for x in range(0,w):
        
            #If the pixel value is above 100 set it to white.  If it is below 100 set it to black.
            if (frame[y,x] > 70):
                imgThres[y,x] = 255
            else:
                imgThres[y,x] = 0
    
    return imgThres


# SKIN DETECTION (HSV) (press S)

def skin_detection(frame):
    
        #frame = cv2.rotate(frame, cv2.ROTATE_180)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        min_HSV = np.array([0,50,20], np.uint8)
        max_HSV = np.array([25,180,250], np.uint8)

        skinArea = cv2.inRange(frame, min_HSV, max_HSV)

        # bitwise-AND mask
        skinHSV = cv2.bitwise_and(frame, frame, mask=skinArea)

        # Color conversion
        skinHSV = cv2.cvtColor(skinHSV, cv2.COLOR_HSV2BGR) 
    
        #resize the video window
        return skinHSV


# Convolution function to convolve through the image
# Takes a frame, kernel as arguments and returns the convolved image

def convolve(image, kernel):
    # Get the image dimensions
    image_height = image.shape[0]
    image_width = image.shape[1]
    # Get the kernel dimensions
    kernel_height = kernel.shape[0]
    kernel_width = kernel.shape[1]
    # Empty array for our output image. Size of the input image.
    output = np.zeros((image_height, image_width))
    # All the rows except for the edge pixels.
    for y in range(image_height - kernel_height):
        # All the pixels except for the edge pixels.
        for x in range(image_width - kernel_width):
        # Mat or kernel frame. Part of the image to perform convolution.
            mat = image[y:y+kernel_height, x:x+kernel_width]
            # Perform convolution.
            output[y,x] = int(np.sum(np.multiply(mat, kernel)))
    # If the output has negative numbers clip to 0->255 range.
    if(np.min(output) < 0):
        output = np.clip(output,0,255)
    # return image
    return output
        

# GAUSSIAN BLUR FILTER (press B)

def blurring(frame):
    # gaussian blur kernel
    blur = np.array(([1,2,1],[2,4,2],[1,2,1]),dtype=np.float32)/16
    # filter2d to convolve through the frame
    imgGaussian = cv2.filter2D(frame, -1, blur)

    return imgGaussian

# SHARPENING FILTER (press X)

def sharpen(frame):
    # sharpening kernel
    sharpen = np.array(([0,-1,0],[-1,5,-1],[0,-1,0]),dtype=np.float32)/16
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    convolve_image = convolve(frame, sharpen)

    return convolve_image



# EDGE DETECTION USING SOBEL (press E)

def edge_detection(frame):
    # Gray-Scale conversion
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Noise-reduction
    img = cv2.GaussianBlur(frame,(5,5),0)
    # Sobel filtering
    Sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, 3)
    Sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, 3)
    # Magnitude calculation
    Magnitude = cv2.magnitude(Sobelx,Sobely)
    # Applying threshold 
    th, threshold = cv2.threshold(Magnitude, 70, 255, cv2.THRESH_BINARY)
    # Displaying 
    return threshold 


# FACE DETECTION USING HAAR-CASCADE (press F)

def face_eye_filter(frame):
    face_cascade = cv2.CascadeClassifier('c:/Users/THIS PC/AppData/Local/Programs/Python/Python311/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')
    eyes_cascade = cv2.CascadeClassifier('c:/Users/THIS PC/AppData/Local/Programs/Python/Python311/Lib/site-packages/cv2/data/haarcascade_eye.xml')

    # Load the input image and convert it to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image using the face cascade classifier
    faces = face_cascade.detectMultiScale(gray)

    # Iterate through all the detected faces and draw ellipses around them
    for (x,y,w,h) in faces:
        center = (round(x + w/2), round(y + h/2))  # Calculate the center of the face
        frame = cv2.ellipse(frame, center, (round(w/2), round(h/2)), 0, 0, 360, (255, 0, 0), 4)  # Draw an ellipse around the face
        faceROI = gray[y:y+h,x:x+w]  # Extract the region of interest corresponding to the face
        # Detect eyes in the face region of interest using the eyes cascade classifier
        eyes = eyes_cascade.detectMultiScale(faceROI)
        # Iterate through all the detected eyes and draw circles around them
        for (x2,y2,w2,h2) in eyes:
            eye_center = (x + x2 + w2//2, y + y2 + h2//2)  # Calculate the center of the eye
            radius = int(round((w2 + h2)*0.25))  # Calculate the radius of the circle
            frame = cv2.circle(frame, eye_center, radius, (0, 0, 255 ), 4)  # Draw a circle around the eye

    # Display the final image using matplotlib
    return frame 


# EDGE-CONTOURS DETECTION (press O)

def contour_track(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Apply Canny edge detection    
    edges = cv2.Canny(gray, 100, 200)
    # Find contours and hierarchy
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Loop through the hierarchy array and print the hierarchy relationships
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

    # Show the image with the contours
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
    elif image_filter == SKIN_DETECTION:
        # Skin-detection
        result = skin_detection(frame) 
    elif image_filter == BLUR_FILTER:
        # blur filter
        result = blurring(frame)
    elif image_filter == SHARPEN_FILTER:
        # Sharpen filter
        result = sharpen(frame)   
    elif image_filter == EDGE_DETECTION:
        # sobel filtering
        result = edge_detection(frame)
    elif image_filter == FACE_EYES_FILTER:
        # object tracking using haar-cascade
        result = face_eye_filter(frame)
    elif image_filter == CONTOUR_TRACK:
        # Contour tracking  
        result = contour_track(frame)
    
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
    # Skin-Detection
    elif key == ord('S') or key == ord('s'):
        image_filter = SKIN_DETECTION 
    # Blur-Filter
    elif key == ord('B') or key == ord('b'):
        image_filter = BLUR_FILTER
    # Sharpen-Filter
    elif key == ord('X') or key == ord('x'):
        image_filter = SHARPEN_FILTER
    # Edge-Detection
    elif key == ord('E') or key == ord('e'):
        image_filter = EDGE_DETECTION
    # Object-Detection using haar-cascade
    elif key == ord('F') or key == ord('f'):
        image_filter = FACE_EYES_FILTER
    # Contour tracking
    elif key == ord('O') or key == ord('o'):
        image_filter = CONTOUR_TRACK

    

##################################################################################
# Close the app
cap.release()
cv2.destroyWindow('frame')



