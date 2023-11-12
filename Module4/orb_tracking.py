import cv2
import numpy as np

# Callback function for trackbar (required even if we don't use it)
def nothing(x):
    pass

# Initialize video capture
cap = cv2.VideoCapture(0)


# Create windows for the trackbars and display
cv2.namedWindow('Settings')

# Create Background Subtractor object
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)

# Initialize a frame count
frame_count = 0

# "Train" the background subtractor with the first 500 frames
while frame_count < 500:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Update the background model
    fgmask = fgbg.apply(frame)
    frame_count += 1


# Initialize ORB detector
orb = cv2.ORB_create()

# Create trackbars for erosion and dilation iterations and contour area
cv2.createTrackbar('Erosion Iterations', 'Settings', 0, 10, nothing)
cv2.createTrackbar('Dilation Iterations', 'Settings', 0, 10, nothing)
cv2.createTrackbar('Min Contour Area', 'Settings', 500, 5000, nothing)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Apply background subtraction 
    fgmask = fgbg.apply(frame, learningRate=0)

    # Get current positions of trackbars
    erosion_iter = cv2.getTrackbarPos('Erosion Iterations', 'Settings')
    dilation_iter = cv2.getTrackbarPos('Dilation Iterations', 'Settings')
    min_contour_area = cv2.getTrackbarPos('Min Contour Area', 'Settings')

    # Apply erosion and dilation with trackbar-adjusted iterations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    fgmask = cv2.erode(fgmask, kernel, iterations=erosion_iter)
    fgmask = cv2.dilate(fgmask, kernel, iterations=dilation_iter)

    # Find contours in the foreground mask
    contours, _ = cv2.findContours(fgmask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Detect ORB features in the frame
    keypoints = orb.detect(frame, None)

    # Loop over contours
    for contour in contours:
        if cv2.contourArea(contour) > min_contour_area:
            # Get the bounding box coordinates around the contour
            x, y, w, h = cv2.boundingRect(contour)
            # Draw the bounding box on the original frame
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Create a mask for the current bounding box to find keypoints within it
            bbox_mask = np.zeros_like(fgmask)
            cv2.rectangle(bbox_mask, (x, y), (x + w, y + h), 255, -1)
            keypoints_bbox = [kp for kp in keypoints if bbox_mask[int(kp.pt[1]), int(kp.pt[0])] == 255]
            frame = cv2.drawKeypoints(frame, keypoints_bbox, frame, color=(255, 0, 0))

    # Display the original frame with the bounding box and keypoints
    cv2.imshow('Frame', frame)

    # Display the foreground mask
    cv2.imshow('Foreground mask', fgmask)

    key = cv2.waitKey(1)
    if key == ord('q') or key == ('Q'):
        break

# Release resources and destroy all windows
cap.release()
cv2.destroyAllWindows()
