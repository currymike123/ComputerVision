# LIEB EVERETT MATHIESON FINAL PROJECT FOR IMAGE-PROCESSING / COMPUTER-VISION
# PLEASE OBSERVE HSV_SKIN AND OTHER FUNCTIONS

# THANK YOU!
##################################################################################
# Project Instructions
"""
Using the provided Python file (final_project_template.py), create an image filtering and tracking app.  Your app needs to have five different key codes that made to filters/tracking.  We will discuss using the python file and the project requirements. 

Your app should have the following:

1. Color Space Conversion or Threshold filter using for loops.
2. HSV Skin detector. (In-class project)
3. Convolution-based filter. (Sharpening, Blurring, etc.)
4. Edge-detecting pipeline with noise reduction (Prewitt/Sobel). 
5. One object detection/tracking algorithm. 
"""

##################################################################################
# Libraries

import cv2
from cv2 import magnitude, Sobel
import numpy as np
import matplotlib.pyplot as plt

##################################################################################
# Variables to map the numeric values to the filter names.  We will use later.
PREVIEW   = 0   # Preview Mode
GRAY      = 1   # Grayscale Filter
THRESHOLD = 2   # Threshold Filter
GLITCH    = 3   # Green-pink glitch using YCrCb conversion
HSV_SKIN  = 4   # DYNAMIC Skin-tone tracker using facial recognition
BLUR      = 5   # Blur Filter
SHARP     = 6   # Sharpen Filter 3X3 kernel
PREWITT   = 7   # Prewitt Filter
SOBEL     = 8   # Sobel Filter
CORNER    = 9   # Shi-Tomasi Corner Detector
FACE_TRAk = 10  # Draw a square around face
ENCODE    = 11  # Encode (compress frame)
YCRCB     = 12  # Decode encoded image 4:2:2 compression standard
YCRCB420  = 13  # Decode encoded image 4:2:0 compression standard
FEELBLUE  = 14  # You're a monster!

##################################################################################
# All your filter and tracker functions.

# Threshold Filter 
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


# Inversion Filter
# Totally useless filter that I accidentally made while being confused in class
def YCBR_Glitch_filter(frame):

    # Add Padding to frame
    # padded_image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    pad_frame = cv2.copyMakeBorder(frame, 2, 2, 4, 4, cv2.BORDER_CONSTANT, value=(0,0,255))

    # Create an empty array of frame size to save color data into
    blended = np.zeros((pad_frame.shape[0],pad_frame.shape[1],pad_frame.shape[2]), dtype=np.uint8)

    # convert frame to YCrCb colorspace for funky colors
    ycrcb = cv2.cvtColor(pad_frame, cv2.COLOR_RGB2YCrCb)

    h = pad_frame.shape[0]
    w = pad_frame.shape[1]

    for y in range(1,h-1):
        for x in range(2,w-2):

            if(y%2 == 0):
                # Save YCrCb color to frame
                blended[y,x] = ycrcb[y,x]
            else:
                blended[y,x,0] = ycrcb[y,x,0]
                blended[y+1,x,1] = ycrcb[y-1,x+1,1]
                blended[y-1,x,2] = pad_frame[y-1,x-2,2]

    return blended


# Dynamic face tracking to determin skin tones based on face region
def HSV_skin(frame):

    # Load face classifier from site-packages
    face_cascade = cv2.CascadeClassifier('/Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages/cv2/data/haarcascade_frontalface_default.xml')

    # make HSV version of the frame
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Convert the frame to grayscale for face detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Perform face detection
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(10, 10))

    # Default in case of no face detection
    face_region_hsv = frame

    # Iterate through detected faces to calculate the average HSV values
    for (x, y, w, h) in faces:
        # Extract the face region from the HSV frame
        face_region_hsv = hsv_frame[y:y+h, x:x+w]

    # Calculate the average HSV values within the region
    avg_hsv = np.mean(face_region_hsv, axis=(0, 1))

    # Calculate the range around the average for lower and upper ranges
    range_hsv = np.std(face_region_hsv, axis=(0, 1))

    # Calculate the lower and upper ranges based on the average and range
    lower_range = np.clip(avg_hsv - range_hsv, 0, 255).astype(np.uint8)
    upper_range = np.clip(avg_hsv + range_hsv, 0, 255).astype(np.uint8)

    # Create a binary mask based on the skin color range
    skin_mask = cv2.inRange(hsv_frame, lower_range, upper_range)

    # Apply the mask to the frame to filter out non-skin regions
    filtered_frame = cv2.bitwise_and(frame, frame, mask=skin_mask)

    return filtered_frame

### Non-dynamic skin tone tracker
# def HSV_skin(frame):
#     # Convert the frame to the HSV color space
#     hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

#     # Define the lower and upper threshold values for skin color in HSV
#     lower_skin = np.array([0, 20, 70], dtype=np.uint8)
#     upper_skin = np.array([20, 255, 255], dtype=np.uint8)

#     # Create a binary mask based on the skin color range
#     skin_mask = cv2.inRange(hsv_frame, lower_skin, upper_skin)

#     # Apply the mask to the frame to filter out non-skin regions
#     filtered_frame = cv2.bitwise_and(frame, frame, mask=skin_mask)

#     return filtered_frame


# Blur Filter
def Blur(frame):
    # Create Home-made Gaussian Filter
    gaussian_filter = np.array([[1,2,1], [2,4,2], [1,2,1]], dtype=np.float32 )/16
    
    # Apply Gaussian
    frameGaussian = cv2.filter2D(frame, -1, gaussian_filter)

    return frameGaussian


# Sharp Filter
def Sharp(frame):
     # Create a sharpening kernel
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])

    # Apply the kernel to the image using filter2D
    frameSharp = cv2.filter2D(frame, -1, kernel)

    return frameSharp


# Prewitt Filter
def Prewitt(frame):

    # Convert to gray
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Create Home-made Gaussian Filter
    gaussian_filter = np.array([[1,2,1], [2,4,2], [1,2,1]], dtype=np.float32 )/16
    
    # Apply Gaussian
    imgGaussian = cv2.filter2D(frame, -1, gaussian_filter)

    # Create X, Y Prewitt Filter Kernels
    kernalX = np.array([[1,1,1], [0,0,0], [-1,-1,-1]], dtype=np.float32)
    kernalY = np.array([[-1,0,1], [-1,0,1], [-1,0,1]], dtype=np.float32)

    # Apply Prewitt Kernels to frame
    framePrewittX = cv2.filter2D(imgGaussian, -1, kernalX)
    framePrewittY = cv2.filter2D(imgGaussian, -1, kernalY)

    # Combine Prewitt Filters into single frame
    Prewitt = cv2.add(framePrewittX, framePrewittY)

    # Return Prewitt Filtered Image
    return Prewitt


# Sobel Filter
def Sobel_filter(frame):

    # Convert to gray
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Noise Reduction
    blur = cv2.GaussianBlur(gray, (3,3),cv2.BORDER_DEFAULT)

    # Height and Width Sobel kernals filter    
    sobelx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)

    # Magnitude Calc : G = SQRT(G^2x + G^2y)
    mag = cv2.magnitude(sobelx, sobely)

    # Threshhold image
    val, thresh = cv2.threshold(mag, 20, 255, cv2.THRESH_BINARY)

    return thresh


# Corner Detector
def Corners(frame):
   
    # Convert to gray
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Convert to float 32 // what the function wants
    gray = np.float32(gray)

    # Perforn Harris Corner Detection
    #dst = cv2.cornerHarris(gray, 2, 5, 0.07)
    dst = cv2.cornerHarris(gray, 6, 9, 0.01)

    # Thresholding
    ret, dst = cv2.threshold(dst, 0.01 * dst.max(), 255, 0)
    dst = np.uint8(dst)

    # Find contours
    contours, hierarchy = cv2.findContours(dst, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw Bounding Boxes around contours
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        rect = cv2.rectangle(frame, (x,y), (x+w, y+h), (10, 255, 15), 1)

    return rect


# Face Tracker
def Face_Track(frame):
    # Load  face classifier from site-packages
    face_cascade = cv2.CascadeClassifier('/Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages/cv2/data/haarcascade_frontalface_default.xml')

    # frame object needs to be converted to gray for face tracking filter to work
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces - Decision Tree
    faces = face_cascade.detectMultiScale(gray)

    # Iterate through all the detected faces and draw rectangle around face region
    for (x,y,w,h) in faces:
        frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    return frame
    

# Encode Filter
def Encode(frame):
    
    # Make B&W version of frame
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # Encode
     # make empty array to size of compressed frame
    blended = np.zeros((frame.shape[0],frame.shape[1],frame.shape[2]), dtype=np.uint8)

     # Loop through frame and create compressed image from gray and color data
    for y in range(0, frame.shape[0]):
        for x in range(0, frame.shape[1]):
            # Use modulo to split every other row
            if(y % 2 == 0):
                # Save color data to even numbered rows.
                blended[y,x] = frame[y,x]
            else:
                # Save B&W data to odd numbered rows
                blended[y,x] = gray[y,x]
    
    # Optional conversion to YCrCb color space
    # blended = cv2.cvtColor(blended, cv2.COLOR_RGB2YCrCb)

    return blended


# YCrCb Filter in 4:2:2 standard
def YCrCb(frame):
    # Import encoded frame
    encoded = Encode(frame)

    # Convert compressed image to YCrCb
    blended = cv2.cvtColor(encoded, cv2.COLOR_RGB2YCrCb)

    # Create blank array to size of compressed image
    decode = np.zeros((frame.shape[0],frame.shape[1],frame.shape[2]), dtype=np.uint8)

    for y in range(0, frame.shape[0]):
        for x in range(0, frame.shape[1]):
            # Use modulo to separate out rows evenly
            if(y % 2 == 0):
                decode[y,x] = blended[y,x]
            else:
                # Save Brightness values and take the change in red/blue from the next row
                decode[y,x,0] = blended[y,x,0]   # Brightness information
                decode[y,x,1] = blended[y-1,x,1] # Red information
                decode[y,x,2] = blended[y-1,x,2] # Blue information
   
    # Importantly convert YCrCb back to RGB color space
    decode = cv2.cvtColor(decode, cv2.COLOR_YCrCb2RGB)
    return decode


# YCrCb Filter in 4:2:0 standard
def YCrCb_420(frame):
    # Import encoded frame
    encoded = Encode(frame)

    # Convert compressed image to YCrCb
    blended = cv2.cvtColor(encoded, cv2.COLOR_RGB2YCrCb)

    # Create blank array to size of compressed image
    decode = np.zeros((frame.shape[0], frame.shape[1], frame.shape[2]), dtype=np.uint8)
    
    # Separate out rows and columns evenly by skipping every other row and columns
    for y in range(0, frame.shape[0], 2):
        for x in range(0, frame.shape[1], 2):
            
            # Save color information for current pixel
            decode[y, x] = blended[y, x]

            # Calculate the values of neighboring pixels for Cr and Cb channels
            decode[y+1,x,0] = blended[y+1,x,0]       # brightness information
            decode[y+1,x,1] = blended[y,x,1]         # Red information
            decode[y+1,x,2] = blended[y,x,2]         # Blue information

            decode[y,x+1,0] = blended[y,x+1,0]       # brightness information
            decode[y,x+1,1] = blended[y,x,1]         # Red information
            decode[y,x+1,2] = blended[y,x,2]         # Blue information

            decode[y+1,x+1,0] = blended[y+1,x+1,0]   # brightness information
            decode[y+1,x+1,1] = blended[y,x,1]       # Red information
            decode[y+1,x+1,2] = blended[y,x,2]       # Blue information

    # Convert YCrCb back to RGB color space
    decoded_frame = cv2.cvtColor(decode, cv2.COLOR_YCrCb2RGB)

    return decoded_frame


# Dynamic face tracking to determin skin tones based on face region and change them to BLUE!
def Blue_skin(frame):

    # Load face classifier from site-packages
    face_cascade = cv2.CascadeClassifier('/Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages/cv2/data/haarcascade_frontalface_default.xml')

    # make HSV version of the frame
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Convert the frame to grayscale for face detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Perform face detection
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(10, 10))

    # Default in case of no face detection
    face_region_hsv = frame

    # Iterate through detected faces to calculate the average HSV values
    for (x, y, w, h) in faces:
        # Extract the face region from the HSV frame
        face_region_hsv = hsv_frame[y:y+h, x:x+w]

    # Calculate the average HSV values within the region
    avg_hsv = np.mean(face_region_hsv, axis=(0, 1))

    # Calculate the range around the average for lower and upper ranges
    range_hsv = np.std(face_region_hsv, axis=(0, 1))

    # Calculate the lower and upper ranges based on the average and range
    lower_range = np.clip(avg_hsv - range_hsv, 0, 255).astype(np.uint8)
    upper_range = np.clip(avg_hsv + range_hsv, 0, 255).astype(np.uint8)

    # Create a binary mask based on the skin color range
    skin_mask = cv2.inRange(hsv_frame, lower_range, upper_range)

    # Apply the mask to the frame to filter out non-skin regions
    filtered_frame = cv2.bitwise_and(frame, frame, mask=skin_mask)

    # Turn you blue
    filtered_frame = cv2.cvtColor(filtered_frame, cv2.COLOR_RGB2BGR)

    # result = cv2.add(frame, filtered_frame)
    # Add blue result to frame, replace normal skin tones
    result = np.where(filtered_frame != 0, filtered_frame, frame)

    return result





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
    elif image_filter == GLITCH:
        # Send frame to glitch function.
        result = YCBR_Glitch_filter(frame)
    elif image_filter == HSV_SKIN:
        # Track face and filter out skin tones in frame.
        result = HSV_skin(frame)
    elif image_filter == BLUR:
        # Blur frame.
        result = Blur(frame)
    elif image_filter == SHARP:
        # Sharpen frame.
        result = Sharp(frame)
    elif image_filter == PREWITT:
        # Send frame to Prewitt filter.
        result = Prewitt(frame)
    elif image_filter == SOBEL:
        # Send frame to Sobel function.
        result = Sobel_filter(frame)
    elif image_filter == CORNER:
        # Detect corners in frame.
        result = Corners(frame)
    elif image_filter == FACE_TRAk:
        # Track faces in frame.
        result = Face_Track(frame)
    elif image_filter == ENCODE:
        # Send frame to encoder.
        result = Encode(frame)
    elif image_filter == YCRCB:
        # Send frame to YCrCb 4:2:2 decoder.
        result = YCrCb(frame)
    elif image_filter == YCRCB420:
        # Send frame to YCrCb 4:2:2 decoder.
        result = YCrCb_420(frame)
    elif image_filter == FEELBLUE:
        # Send frame to Sobel function.
        result = Blue_skin(frame)
    
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
    # Glitch Filter
    elif key == ord('I') or key == ord('i'):
        image_filter = GLITCH
    # Encode image for YCbCr Filter
    elif key == ord('E') or key == ord('e'):
        image_filter = ENCODE
    # YCrCb Filter
    elif key == ord('Y') or key == ord('y'):
        image_filter = YCRCB
    # YCrCb in 4:2:0 Filter
    elif key == ord('U') or key == ord('u'):
        image_filter = YCRCB420
    # Sharp Filter
    elif key == ord('W') or key == ord('w'):
        image_filter = SHARP 
    # Blur Filter
    elif key == ord('B') or key == ord('b'):
        image_filter = BLUR    
    # Sobel Filter
    elif key == ord('S') or key == ord('s'):
        image_filter = SOBEL
    # Prewitt Filter
    elif key == ord('K') or key == ord('k'):
        image_filter = PREWITT
    # Corner Detection Filter
    elif key == ord('C') or key == ord('c'):
        image_filter = CORNER
    # Face Tracking Filter
    elif key == ord('F') or key == ord('f'):
        image_filter = FACE_TRAk
    # Skin HSV filter
    elif key == ord('H') or key == ord('h'):
        image_filter = HSV_SKIN
    elif key == ord('M') or key == ord('m'):
        image_filter = FEELBLUE

##################################################################################
# Close the app
cap.release()
cv2.destroyWindow('frame')



