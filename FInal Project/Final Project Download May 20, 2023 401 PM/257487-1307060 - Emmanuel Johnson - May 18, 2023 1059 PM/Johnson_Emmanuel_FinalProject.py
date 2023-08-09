##################################################################################
# Libraries

import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse

#1. Color Space Conversion or Threshold filter using for loops
#2. HSV Skin detector. (In-class project)
#3. Convolution-based filter. (Sharpening, Blurring, etc.)
#4. Edge-detecting pipeline with noise reduction (Prewitt/Sobel)
#5. One object detection/tracking algorithm

##################################################################################
# Variables to map the numeric values to the filter names.  We will use later.
PREVIEW   = 0   # Preview Mode
GRAY      = 1   # Grayscale Filter
THRESHOLD = 2   # Threshold Filter
SKINDETECTOR = 3 # HSV Skin Detector
NOISEREDUCT = 4 # Noise Reduction
SHARPEN = 5 # Noise Reduction
BLUR = 6 # Noise Reduction
EDGEDETECTOR = 7 # Edge Detection with noise reduction
EDGE = 8 # Edge Detection with noise reduction
OBJECTDETECTION = 9 # Object Detection


##################################################################################
# All your filter and tracker functions.

# Color Space & Threshold Filter
def Threshold_filter(frame):
    # Color Space Conversion
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    h = frame.shape[0]
    w = frame.shape[1]
    
    # Thresholding Using For Loops
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

def MakeGray(frame): 
    #Grayscale the frame
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame

# HSV Skin detector. (In-class project)
def HSVSkinDetector(frame):
    #Grayscale the frame
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # # Minimum and maximum HSV values.
    min_HSV = np.array([0,100,50], np.uint8)
    max_HSV = np.array([25,175,255], np.uint8)
    
    # # cv2.inRange(image, minimum, maximum)
    skinArea = cv2.inRange(frame, min_HSV, max_HSV)
    
    # # Bitwise And mask
    skinHSV = cv2.bitwise_and(frame, frame, mask=skinArea)

    # # Convert to RGB
    skinHSV = cv2.cvtColor(skinHSV, cv2.COLOR_HSV2BGR)
    
    return skinHSV

# Convolution-based filter. (Sharpening, Blurring, etc.)

def Sharpening(frame):
    
    #Grayscale the frame
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    sharpen = np.array(([0,-1,0],
                    [-1,5,-1],
                    [0,-1,0]),
                    dtype=np.float32)
    
    imgConv2 = cv2.filter2D(frame,-1, sharpen) #alternative, doesn't work convolve_image = int(np.sum(np.multiply(frame, kernel)))
    return imgConv2

def Blurring(frame):
    
    #Grayscale the frame
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    blur = np.array(([.111,.111,.111],
                   [.111,.111,.111],
                   [.111,.111,.111]),
                   dtype=np.float32)
    
    imgConv2 = cv2.filter2D(frame,-1, blur)
    
    return imgConv2

def Edge(frame):
    
    #Grayscale the frame
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    edge = np.array(([-1,-1,-1],
                 [-1,8,-1],
                 [-1,-1,-1]),
                 dtype=np.float32)
    
    imgConv2 = cv2.filter2D(frame,-1, edge)
    
    return imgConv2
    
def NoiseReduction(frame):
    
    #Grayscale the frame
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    #2. Noise Reduction: Gaussian filter 
    gaussian_filter = np.array([[1,2,1],[2,4,2],[1,2,1]], dtype=np.float32) / 16

    # Convolve with the image
    imgGaussian = cv2.filter2D(frame, -1, gaussian_filter)

    # Plot the combined image
    #fig = plt.figure(figsize = (15,15)); plt.imshow(frame,cmap='gray')
    
    return imgGaussian

def SharpeningManual(frame): # Sharpening Done With Loops
    
    #Grayscale the frame
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    
    # Get the image dimensions
    image_height = frame.shape[0]
    image_width = frame.shape[1]
    
    #sharpening kernel
    sharpen = np.array(([0,-1,0],
                    [-1,5,-1],
                    [0,-1,0]),
                    dtype=np.float32)
    
    # Get the kernel dimensions
    kernel_height = sharpen.shape[0]
    kernel_width = sharpen.shape[1]
    
    # Empty array for our output image. Size of the input image.
    output = np.zeros((image_height, image_width))
    
    # All the rows except for the edge pixels.
    for y in range(image_height - kernel_height):
        # All the pixels except for the edge pixels.
        for x in range(image_width - kernel_width):

            # Mat or kernel frame. Part of the image to perform convolution.
            mat = frame[y:y+kernel_height, x:x+kernel_width]

            # Perform convolution.
            output[y,x] = int(np.sum(np.multiply(mat, sharpen)))
            
    # If the output has negative numbers clip to 0->255 range.
    if(np.min(output) < 0):
        output = np.clip(output,0,255)
    
    return output

# Edge-detecting pipeline with noise reduction (Prewitt/Sobel)
def EdgeDetection(frame):
    
    frame = MakeGray(frame) #Grayscale the frame
    frame = NoiseReduction(frame) # noise reduction with Gaussian filter
    kernelX = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)
    kernelY = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)

    # convolution
    #imgSobelX = cv2.filter2D(frame, -1, kernelX) #frame should be from imgGaussian filter (noise reduction)
    #imgSobelY = cv2.filter2D(frame, -1, kernelY) 
    #imgSobelXY = cv2.add(imgSobelX, imgSobelY) #or cv2.magnitude(imgSobelX,imgSobelY)

    dx = cv2.Sobel(frame, cv2.CV_32F, 1, 0)  # float ; frame should be from imgGaussian filter
    dy = cv2.Sobel(frame, cv2.CV_32F, 0, 1)
    
    mag = cv2.magnitude(dx,dy)
    
    # Thresholding; cmap=gray if possible
    th, imgOcvThres = cv2.threshold(mag, 100, 255, cv2.THRESH_BINARY) # could replace mag with imgSobelXY
    
    return imgOcvThres

# One object detection/tracking algorithm
def HaarObjectDetection(frame):
    
    # Load the face and eyes cascade classifiers
    #ok = cv2.imread() # works but returns None as it's an invalid file name

    #face_cascade = cv2.CascadeClassifier('Assignments/haarcascade_eye.xml')
    #eyes_cascade = cv2.CascadeClassifier('Assignments/haarcascade_eye.xml')
    #/Users/emmanueljohnson/Documents/Schoolwork New Paltz/NP 22-23/SPRING 2023/Image Processing/Assignments/Assignemnt/lib/python3.10/site-packages/cv2/data/haarcascade_eye.xml
    #/cv2/data/haarcascade_frontalface_default.xml
    
    face_cascade = cv2.CascadeClassifier()
    eyes_cascade = cv2.CascadeClassifier()
    
    face_cascade.load(cv2.samples.findFile("haarcascade_frontalface_default.xml"))
    eyes_cascade.load(cv2.samples.findFile("haarcascade_eye.xml"))
    
    # Load the input image and convert it to grayscale
    gray = MakeGray(frame)
    frame = MakeGray(frame)
    

    # Detect faces in the grayscale image using the face cascade classifier
    faces = face_cascade.detectMultiScale(frame)

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

##################################################################################
# The video image loop.

# Create a VideoCapture object and read from input file
# If the input is taken from the camera, pass 0 instead of the video file name.
cap = cv2.VideoCapture(2)

# Variable to keep track of the current image filter. Default set to Preview.
image_filter = EDGE

# Video Loop
while cap.isOpened():
    # Ret = True if frame is read correctly, frame = the frame that was read.
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        break
    
    # Resize the image so your app runs faster.  Less Pixels to Process.  
    # Use Nearest Neighbor Interpolation for the fastest runtime.
    frame = cv2.resize(frame, None, fx = 0.3, fy = 0.3, interpolation=cv2.INTER_NEAREST) #cv2.ROTATE_180
    
    # Send your frame to each filter.
    if image_filter == PREVIEW:
        # No filter.
        result = frame
    elif image_filter == THRESHOLD:
        # Send the frame to the Threshold function.
        result = Threshold_filter(frame)
    elif image_filter == GRAY:
        # Convert to grayscale.
        result = MakeGray(frame)
    elif image_filter == SKINDETECTOR:
        # Send to HSV Skin Detector Function.
        result = HSVSkinDetector(frame)
    elif image_filter == NOISEREDUCT:
        # Send to Noise Reduction Function.
        result = NoiseReduction(frame)
    elif image_filter == EDGEDETECTOR:
        # Send to Edge Detection Function.
        result = EdgeDetection(frame)
    elif image_filter == SHARPEN:
        # Send to Sharpening Function.
        result = Sharpening(frame)
    elif image_filter == BLUR:
        # Send to Blur Function.
        result = Blurring(frame)
    elif image_filter == EDGE:
        # Send to Secondary Edge Detection Function.
        result = Edge(frame)
    elif image_filter == OBJECTDETECTION:
        # Send to Haar Object Detection
        result = HaarObjectDetection(frame)

    
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
    # HSV Skin Detector filter
    elif key == ord('H') or key == ord('h'):
        image_filter = SKINDETECTOR
    # Edge Detection filter - works when typed in manually, Crashing on key press
    elif key == ord('F') or key == ord('f'): #THIS IS ACTUALLY THE 'E' KEY ON MAC
        image_filter = EDGEDETECTOR
    # Blur filter
    elif key == ord('B') or key == ord('b'):
        image_filter = BLUR
    # Sharpen filter
    elif key == ord('S') or key == ord('s'):
        image_filter = SHARPEN
    # Edge filter
    elif key == ord('M') or key == ord('e'):
        image_filter = EDGE
    # Object Detection filter
    elif key == ord('O') or key == ord('o'):
        image_filter = OBJECTDETECTION

##################################################################################
# Close the app
cap.release()
cv2.destroyWindow('frame')



