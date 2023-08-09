##################################################################################
# Libraries

import cv2
import numpy as np
import matplotlib.pyplot as plt
import copy

##################################################################################
# Variables to map the numeric values to the filter names.  We will use later.
PREVIEW   = 0   # Preview Mode            ' KEY P'
GRAY      = 1   # Grayscale Filter        ' KEY G'
THRESHOLD = 2   # Threshold Filter        ' KEY T'
HSV       = 3   # RGB to HSV              ' KEY H'
SKIN      = 4   # skin detection          ' KEY S'
BLUR      = 5   # blur the image by converting it to grayscale         ' KEY B'
SHARP     = 6   # sharpen the image by converting it to grayscale      ' KEY A'
EDGE      = 7   # Sobel Edge               ' KEY E'
FACE      = 8   # Face Detection           ' KEY F'

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

#RGB to HSV 

def hsv_filter(img):
    frame = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return frame
  

# SKIN DETECTION 

def skin_detection(frame):
    
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


# CONVOLVE function For BLUR and SHARP 
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


#Blur function
def blur(frame):


    #img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img=frame
    imgGaussian=cv2.GaussianBlur(img,(7,7),0)

    return imgGaussian


#sharpern Function

def sharp(frame):


    #img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img=frame

    sharpen = np.array(([0,-1,0],[-1,5,-1],[0,-1,0]),dtype=np.float32)
    imgConv2 = cv2.filter2D(img,-1,sharpen)

    return imgConv2

#  Edge detion using Sobel 

def sobel_edge(frame):
    img=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    imgGaussian=cv2.GaussianBlur(img,(7,7),0)

    #Sobel convolution
    imgSobelX = cv2.Sobel(imgGaussian, cv2.CV_64F, 1,0)
    imgSobelY = cv2.Sobel(imgGaussian, cv2.CV_64F, 0,1)

    imgSobel=cv2.magnitude(imgSobelX,imgSobelY)

    retval,dst=cv2.threshold(imgSobel,100,255,cv2.THRESH_BINARY)

    return dst


# FACE RECOGNITION

def face_recognition(frame):
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
    frame = cv2.resize(frame, None, fx = 0.5, fy = 0.5, interpolation=cv2.INTER_NEAREST)
    
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
        # Convert to hsv
        result = hsv_filter(frame)

    elif image_filter == SKIN:
        # skin detection
        result = skin_detection(frame)

    elif image_filter == BLUR:
        # blurring image
        result = blur(frame)

    elif image_filter == SHARP:
        # sharpening image
        result = sharp(frame)

    elif image_filter == EDGE:
        # edge detection using sobel in image
        result = sobel_edge(frame)

    elif image_filter == FACE:
        # Face Detection 
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
    # HSV filter
    elif key == ord('H') or key == ord('h'):
        image_filter = HSV
    # Skin Detection
    elif key == ord('S') or key == ord('s'):
        image_filter = SKIN
    # Blur the image
    elif key == ord('B') or key == ord('b'):
        image_filter = BLUR
    # Sharpen the image
    elif key == ord('A') or key == ord('a'):
        image_filter = SHARP
    # Find Edges in image using Sobel
    elif key == ord('E') or key == ord('e'):
        image_filter = EDGE
    # Identify eyes and face
    elif key == ord('F') or key == ord('f'):
        image_filter = FACE


    elif key == ord('P') or key == ord('p'):
        image_filter = PREVIEW
    

##################################################################################
# Close the app
cap.release()
cv2.destroyWindow('frame')



