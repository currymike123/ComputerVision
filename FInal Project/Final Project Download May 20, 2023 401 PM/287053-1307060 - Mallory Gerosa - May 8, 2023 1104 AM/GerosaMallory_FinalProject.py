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
SKINDETECT = 3  # HSV skin detection 
PREWITT = 4     # Prewitt edge detecting
GAUSSIAN = 5    # Gaussian blur
SOBEL = 6       # Sobel filter
HAAR = 7        # Haar Cascade filter

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

def SkinDetect_filter(frame):
    min_HSV = np.array([0,30,0], np.uint8)
    max_HSV = np.array([255,170,240], np.uint8)
    skinArea = cv2.inRange(frame, min_HSV, max_HSV)
    skinHSVFrame = cv2.bitwise_and(frame, frame, mask=skinArea)
    return skinHSVFrame

def GaussianBlur_filter(frame):
    gaussian_filter_kernel = np.array([[1,2,1],[2,4,2],[1,2,1]], dtype=np.float32) / 16
    gaussianBlurFrame = cv2.filter2D(frame, -1, gaussian_filter_kernel)
    return gaussianBlurFrame

def Prewitt_filter(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  
    kernelX = np.array([[1,1,1],[0,0,0],[-1,-1,-1]], dtype=np.float32)
    kernelY = np.array([[-1,0,1],[-1,0,1],[-1,0,1]], dtype=np.float32)
    imgPrewittX = cv2.filter2D(frame, -1, kernelX)
    imgPrewittY = cv2.filter2D(frame, -1, kernelY)
    prewittFrame = cv2.add(imgPrewittX, imgPrewittY)
    return prewittFrame

def Sobel_filter(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gaussian_filter = np.array([[1,2,1],[2,4,2],[1,2,1]], dtype=np.float32) / 16
    frame = cv2.filter2D(frame, -1, gaussian_filter)
    blur = cv2.GaussianBlur(frame, (3,3), 0)
    sobely=cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
    sobelx=cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
    magn = cv2.magnitude(sobelx, sobely)
    th, imgOcvThres = cv2.threshold(magn, 100, 255, cv2.THRESH_BINARY)
    return imgOcvThres

def HaarCascade_filter(frame):
    face_cascade = cv2.CascadeClassifier(r"C:\Users\19144\AppData\Local\Programs\Python\Python311-32\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml")
    eyes_cascade = cv2.CascadeClassifier(r"C:\Users\19144\AppData\Local\Programs\Python\Python311-32\Lib\site-packages\cv2\data\haarcascade_eye.xml")
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)

    for (x,y,w,h) in faces:
        center = (round(x + w/2), round(y + h/2))
        frame = cv2.ellipse(frame, center, (round(w/2), round(h/2)), 0, 0, 360, (255, 0, 0), 4) 
        faceROI = gray[y:y+h,x:x+w]  
        eyes = eyes_cascade.detectMultiScale(faceROI)
        for (x2,y2,w2,h2) in eyes:
            eye_center = (x + x2 + w2//2, y + y2 + h2//2)  
            radius = int(round((w2 + h2)*0.25))  
            frame = cv2.circle(frame, eye_center, radius, (0, 0, 255 ), 4) 
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
    elif image_filter == GRAY:
        # Grayscale filter
        result = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)   
        # Threshold filter 
    elif image_filter == THRESHOLD:
        result = Threshold_filter(frame)
        # HSV skin tracking filter
    elif image_filter == SKINDETECT:
        result = SkinDetect_filter(frame)
    elif image_filter == PREWITT:
        # Send the frame to the HSV skin detecting function
        result = Prewitt_filter(frame)
    elif image_filter == GAUSSIAN:
        # Send the frame to the HSV skin detecting function
        result = GaussianBlur_filter(frame)  
    elif image_filter == SOBEL:
        result = Sobel_filter(frame)
    elif image_filter == HAAR:
        result = HaarCascade_filter(frame)
    # Show your frame.  Remember you don't need to convert to RGB.  
    cv2.imshow('frame', result)
    
    # Map all your key codes to your filters/trackers. Capital and lower case.
    key = cv2.waitKey(1)

    # Quit the app
    if key == ord('Q') or key == ord('q'):
        break
    # Preview. No filter
    elif key == ord('P') or key == ord('p'):
        image_filter = PREVIEW    
    # Grayscale filter
    elif key == ord('G') or key == ord('g'):
        image_filter = GRAY
    # Threshold filter
    elif key == ord('T') or key == ord('t'):
        image_filter = THRESHOLD
    # HSV skin detection filter
    elif key == ord('H') or key == ord('h'):
        image_filter = SKINDETECT
    # Prewitt filter
    elif key == ord('W') or key == ord('w'):
        image_filter = PREWITT
    # Gaussian filter
    elif key == ord('B') or key == ord('b'):
        image_filter = GAUSSIAN
    # Sobel filter
    elif key == ord('S') or key == ord('s'):
        image_filter = SOBEL
    # Haar cascade filter
    elif key == ord('C') or key == ord('c'):
        image_filter = HAAR
##################################################################################
# Close the app
cap.release()
cv2.destroyWindow('frame')



