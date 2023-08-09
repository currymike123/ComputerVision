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
BLUR      = 3   # blur filter
SKINMASK  = 4   # skin masking
OBJECTDETECTION = 5  # object detection
RGBTOHSV  = 6       #hsv filter
EDGEDETECTION = 7   #edge detector w/sobel
SHARPEN   = 8   # sharpen filter
EMBOSS    = 9   #emboss filter
##################################################################################
# All your filter and tracker functions. Add your own here including a convolution filter (blurring sharpening etc.), 
# edge detecting/ noise reduction, object detection, and skin detector 
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


def convolve(image,kernal): # so im sure you wanted us to use the hand written convolution function, I tried but it was incredibly slow and froze a lot
                            #troubleshooted with just doing gray/image compression and accounting for the color channels neither fixed the issue
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    image_height = image.shape[0]
    image_width= image.shape[1]
    

    kernal_height = kernal.shape[0]
    kernal_width = kernal.shape[1]


    output = np.zeros((image_height,image_width))

    for y in range(image_height-kernal_height):
        for x in range(image_width-kernal_width):

            mat = image[y:y+kernal_height,x:x+kernal_width]#taking the pixels within the kernal frame 
            output[y,x] = int(np.sum((np.multiply(mat,kernal)))) #add them to output array after calculating values
           
            # Tried accounting for the color channels but this did not solve the issue
            #for c in range(num_channels):
                # Perform convolution for the current channel
             #   output[y, x, c] = np.sum(np.multiply(mat[:, :, c], kernel))
                # If the output has negative numbers or exceeds 255, clip to the 0-255 range
            #  output[y, x, c] = np.clip(output[y, x, c], 0, 255)
    
    if(np.min(output)<0):
        output = np.clip(output, 0,255)
    
    
    return output
  

def Gausian_blur(frame): # due to the convolution function freezing so much I used filter2D instead for the image enhancements

    blur = np.array(([.111,.111,.111],
                    [.111,.111,.111],
                    [.111,.111,.111]),
                    dtype=np.float32)
    
  
    imgGaussian = cv2.filter2D(src=frame, ddepth=-1, kernel=blur)

    return imgGaussian

def sharpening(frame):

    sharpen = np.array(([0,-1,0],[-1,5,-1],[0,-1,0]),dtype=np.float32)
    sharpened = cv2.filter2D(src=frame,ddepth=-1,kernel=sharpen)
    
    return sharpened

def emboss(frame):

    emboss = np.array(kernel = np.array([
                            [-2, -1, 0],
                            [-1, 1, 1],
                            [0, 1, 2]
                            ]),dtype=np.float32)
    
    embossed = cv2.filter2D(src=frame, ddepth=-1, kerknel=emboss)

    return embossed

def skin_Tone_Masking(frame):

    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    frame = cv2.cvtColor(frame,cv2.COLOR_RGB2HSV)

    min_HSV = np.array([0,20,170], np.uint8)
    max_HSV = np.array([90,180,255],np.uint8)

    skinArea = cv2.inRange(frame, min_HSV, max_HSV)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    
    #considered not using morphology at all- I kinda think it works better without it- it was hard to find a good balance between
    # erode and dilate and worked betteer with just dilate

    #skinArea = cv2.erode(skinArea, kernel, iterations = 1)
    skinArea = cv2.dilate(skinArea, kernel, iterations = 1)
    #bitwise returns a 1 or 0 based off the mask to decide where to mask and where not to
    skinHSV = cv2.bitwise_and(frame, frame, mask=skinArea)
    skinHSV = cv2.cvtColor(skinHSV, cv2.COLOR_HSV2BGR)

    return skinHSV;

def object_detection(frame):
    
   # load in your cascade classifiers (wherever they are on your computer
   # I had to download the opencv master file for the cascade xmls and put them directly in my programs folder to access (was having issues otherwise)
    face_cascade = cv2.CascadeClassifier("opencv-master\data\haarcascades\haarcascade_frontalface_default.xml")
    eye_cascade = cv2.CascadeClassifier("opencv-master\data\haarcascades\haarcascade_eye.xml") 
   
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray)
   
    for (x,y,w,h) in faces:
        center = (round(x + w/2), round(y + h/2))       
        radius = int(round((w + h)*0.25)) 
        frame = cv2.circle(frame, center, radius, (255, 0, 0), 4) # this is all pulled from the sample code posted
        faceROI = gray[y:y+h,x:x+w] 
       
        eyes = eye_cascade.detectMultiScale(faceROI)
        
        for (x2,y2,w2,h2) in eyes:
            eye_center = (x + x2 + w2//2, y + y2 + h2//2) 
            radius = int(round((w2 + h2)*0.25)) 
            frame = cv2.circle(frame, eye_center, radius, (0, 0, 255), 4) 


    return frame

def HSV_color_coversion(frame):  # I know this is supposed to be slow, I tried to add image compression but since its frame
                                 # by frame its just as slow with image compression and resizing                 
    height = frame.shape[0]
    w = frame.shape[1]
    hsv = np.zeros((height,w,3),dtype=np.uint8) #create hsv array structure
        
    for y in range(0,height): #looping through all pixels
        for x in range(0,w):
            
            r = frame[y,x,0]/255
            g= frame[y,x,1]/255 #converting to num between 0-1
            b = frame[y,x,2]/255

                    
            v = max(r,g,b)#get the max from the three values which is assigned to v
            vmin = min(r,g,b)#get the min from the three values
            diff=(v-vmin)
            
            if(v> 0.0):
                s = diff/v #calculate s
            else: 
                s= 0.0
            

            if(r==g and g==b): #if all values are equal then h is 0
                h=0
            elif(r==v): #if red is the max value 
                h=60*(g-b)/diff
            elif(g==v): #if green is the max value
                h=120+60*(b-r)
            elif(b==v): #if blue is the max value
                h=240+60*(r-g)/diff

            ns = np.interp(s,[0,1],[0,255]) #convert s and v to openCV
            nv = np.interp(v,[0,1],[0,255])
           
            # clip values to valid range if necessary since I was getting out of bound values
            if(h,ns,nv>255 or h,ns,nv<255):
                h = np.clip(h, 0, 255)
                ns = np.clip(ns, 0, 255)
                nv = np.clip(nv, 0, 255)

            hsv[y,x]=[round(h/2),round(ns),round(nv)] #add pixels to hsv array no need to add to 3rd placement since its not changing

            

    return hsv 
 

def canny_detection(frame): # did not end up using 

    edges = cv2.Canny(frame, 100, 200)
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    outer_contours = []
    for i in range(len(hierarchy[0])): #iterate through edges
     
        if hierarchy[0][i][3] == -1: # if contour has no parent then its an outter contour
            outer_contours.append(contours[i])

    for cnt in outer_contours: # draw border box around outer contours
        x,y,w,h = cv2.boundingRect(cnt)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),1)

    return frame

def edge_detection(frame): # sobel operator edge detector with noise reduction

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 

    blurred = cv2.GaussianBlur(frame, (3,3),0)

    sobelx = cv2.Sobel(blurred,cv2.CV_32F,1,0,ksize=3)#calc difference in intensity
    sobely = cv2.Sobel(blurred,cv2.CV_32F,0,1,ksize=3)#for both x and y

    magnitude = cv2.magnitude(sobelx,sobely)
    img_thresh = cv2.threshold(magnitude, 100, 255, cv2.THRESH_BINARY)

    return img_thresh[1]
    
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

    # then add in object detection
    # defining edges 
    # add in filters 

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
    elif image_filter == BLUR:
        result = Gausian_blur(frame)
    elif image_filter == SKINMASK:
        result = skin_Tone_Masking(frame)
    elif image_filter == OBJECTDETECTION:
        result = object_detection(frame)
    elif image_filter == RGBTOHSV:
        result = HSV_color_coversion(frame) 
    elif image_filter == EDGEDETECTION:
        result = edge_detection(frame)
    elif image_filter == SHARPEN:
        result = sharpening(frame)   
    elif image_filter == EMBOSS:
        result = emboss(frame)                    
            
     

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
    elif key == ord('B') or key == ord('b'):
        image_filter = BLUR
    elif key == ord('m') or key == ord('S'):
        image_filter = SKINMASK
    elif key == ord('o') or key == ord('O'):
        image_filter = OBJECTDETECTION
    elif key == ord('h') or key == ord('H'):
        image_filter = RGBTOHSV
    elif key == ord('e') or key == ord("E"):
        image_filter = EDGEDETECTION  
    elif key == ord('s') or key == ord('S'):
        image_filter = SHARPEN
    elif key == ord('e') or key == ord('E'):
        image_filter = EMBOSS

##################################################################################
# Close the app
cap.release()
cv2.destroyWindow('frame')



