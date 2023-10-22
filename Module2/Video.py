### Video Loop

import cv2
import numpy as np

####################################################################

# Filter Variables to numeric values.
PREVIEW = 0
GRAY = 1
THRESHOLD = 2
Sobel1 = 3
Sobel2 = 4
Sobel3 = 5
Sobel4 = 6
Sobel5 = 7
Sobel6 = 8
Object = 10


####################################################################

sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
sobel_kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)

def Sobel1(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return gray

def Sobel2(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    return blur

def Sobel3(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    sobelx = cv2.filter2D(blur, cv2.CV_64F, sobel_kernel_x)
    return sobelx

def Sobel4(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    sobelx = cv2.filter2D(blur, cv2.CV_64F, sobel_kernel_x)
    sobely = cv2.filter2D(blur, cv2.CV_64F, sobel_kernel_y)
    img = cv2.add(sobelx,sobely)
    return img

def Sobel5(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    sobelx = cv2.filter2D(blur, cv2.CV_64F, sobel_kernel_x)
    sobely = cv2.filter2D(blur, cv2.CV_64F, sobel_kernel_y)
    sobelx = sobelx.astype(np.float64)
    sobely = sobely.astype(np.float64)
    mag = cv2.magnitude(sobelx, sobely)
    return mag

def Sobel6(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    sobelx = cv2.filter2D(blur, cv2.CV_64F, sobel_kernel_x)
    sobely = cv2.filter2D(blur, cv2.CV_64F, sobel_kernel_y)
    sobelx = sobelx.astype(np.float64)
    sobely = sobely.astype(np.float64)
    mag = cv2.magnitude(sobelx, sobely)
    val, thresh = cv2.threshold(mag, 120, 255, cv2.THRESH_BINARY)
    return thresh



def Threshold(frame):

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    h = frame.shape[0]
    w = frame.shape[1]

    imgThres = np.zeros((h,w), dtype=np.uint8)

    for y in range(0,h):
        for x in range(0,w):

            if(frame[y,x] > 100):
                imgThres[y,x] = 255
            else:
                imgThres[y,x] = 0

    return imgThres

def Object(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply Canny edge detection
    edges = cv2.Canny(blurred,100,220)

    # Find contours
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Find outermost contours using hierarchy
    outer_contours = []
    # Loop through all the contours and their hierarchy information
    for i in range(len(hierarchy[0])):
        # If the contour doesn't have a parent (-1), then it's an outermost contour
        if hierarchy[0][i][3] == -1:
            outer_contours.append(contours[i])

    # Draw bounding box around outermost contours
    for cnt in outer_contours:
        x,y,w,h = cv2.boundingRect(cnt)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),1)

    return frame


####################################################################

# Create a Video Capture Object and read from input file or the webcam.  
# If the input is taken from a camera or the webcam, pass 0.
# If it's a file path and file name.

cap = cv2.VideoCapture(0)

# Variable to keep track of our current image filter. Default to Preview.
image_filter = PREVIEW

# Video loop
while cap.isOpened():

    # Ret = True if frame is read correctly. frame = the current video frame.
    ret, frame = cap.read()

    # If frame is read correctly ret is True
    if not ret:
        break 

    # Resize the frame with Nearest Neighbor Interpolation.
    frame = cv2.resize(frame, None, fx = 0.7, fy = 0.7, interpolation=cv2.INTER_NEAREST)
    
    if image_filter == PREVIEW:
        result = frame
    elif image_filter == THRESHOLD:
        result = Threshold(frame)
    elif image_filter == GRAY:
        result = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    elif image_filter == Sobel1:
        result = Sobel1(frame)
    elif image_filter == Sobel2:
        result = Sobel2(frame)
    elif image_filter == Sobel3:
        result = Sobel3(frame)
    elif image_filter == Sobel4:
        result = Sobel4(frame)
    elif image_filter == Sobel5:
        result = Sobel5(frame)
    elif image_filter == Sobel6:
        result = Sobel6(frame)
    elif image_filter == Object:
        result = Object(frame)
    

    # Show current frame.
    cv2.imshow('frame', result)

    # Map keyboard input to filters/trackers. Capital and lower case.
    key = cv2.waitKey(1)

    # Quit the app.
    if key == ord('Q') or key == ord('q'):
        break
    if key == ord('T') or key == ord('t'):
        image_filter = THRESHOLD
    if key == ord('P') or key == ord('p'):
        image_filter = PREVIEW
    if key == ord('G') or key == ord('g'):
        image_filter = GRAY
    if key == ord('Z') or key == ord('z'):
        image_filter = Sobel1
    if key == ord('X') or key == ord('x'):
        image_filter = Sobel2
    if key == ord('C') or key == ord('c'):
        image_filter = Sobel3
    if key == ord('V') or key == ord('v'):
        image_filter = Sobel4
    if key == ord('B') or key == ord('b'):
        image_filter = Sobel5
    if key == ord('N') or key == ord('n'):
        image_filter = Sobel6
    if key == ord('O') or key == ord('o'):
        image_filter = Object

#Close the app
cap.release()
cv2.destroyWindow('frame')

