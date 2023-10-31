import cv2
import numpy as np
import sys

# Function needed for trackbar creation
def nothing(x):
    pass

# Video capture from file
cap = cv2.VideoCapture("shapes.mp4")

# Check if the video file opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Create a window for the HSV trackbars
cv2.namedWindow("HSV Trackbars")

# Create trackbars for HSV range selection
cv2.createTrackbar("Low-H", "HSV Trackbars", 0, 179, nothing)
cv2.createTrackbar("High-H", "HSV Trackbars", 179, 179, nothing)
cv2.createTrackbar("Low-S", "HSV Trackbars", 0, 255, nothing)
cv2.createTrackbar("High-S", "HSV Trackbars", 255, 255, nothing)
cv2.createTrackbar("Low-V", "HSV Trackbars", 0, 255, nothing)
cv2.createTrackbar("High-V", "HSV Trackbars", 255, 255, nothing)    
cv2.createTrackbar("Min Area", "HSV Trackbars", 0, 5000, nothing) 
cv2.createTrackbar("Max Area", "HSV Trackbars", 5000, 5000, nothing) 


while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # If the frame is not retrieved, then we've reached the end of the video
    if not ret:
        print("Reached the end of the video, looping...")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue
    
    # Resize the frame
    frame = cv2.resize(frame, None, fx = 0.5, fy = 0.5, interpolation=cv2.INTER_NEAREST)

    # Convert the frame to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Get the current positions of the HSV trackbars
    low_h = cv2.getTrackbarPos("Low-H", "HSV Trackbars")
    high_h = cv2.getTrackbarPos("High-H", "HSV Trackbars")
    low_s = cv2.getTrackbarPos("Low-S", "HSV Trackbars")
    high_s = cv2.getTrackbarPos("High-S", "HSV Trackbars")
    low_v = cv2.getTrackbarPos("Low-V", "HSV Trackbars")
    high_v = cv2.getTrackbarPos("High-V", "HSV Trackbars")
    min_area = cv2.getTrackbarPos("Min Area", "HSV Trackbars")
    max_area = cv2.getTrackbarPos("Max Area", "HSV Trackbars")

    # Create a mask using the current HSV range
    color_mask = cv2.inRange(hsv, (low_h, low_s, low_v), (high_h, high_s, high_v))

    # Apply the mask to get the result
    result = cv2.bitwise_and(frame, frame, mask=color_mask)

    # Add erode and dilate to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    #color_mask = cv2.erode(color_mask, kernel, iterations=1)
    #color_mask = cv2.dilate(color_mask, kernel, iterations=1)

    # Convert the result to grayscale (required for Canny)
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection; you might have to adjust the thresholds
    edges = cv2.Canny(gray, 100, 200)

    # Find contours in the edged image.
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw each contour only for visualization purposes
    for contour in contours:
        if min_area < cv2.contourArea(contour) < max_area:    # Updated this conditional
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)


    # Display the original frame with the drawn contours and the mask
    cv2.imshow('Frame with tracked object', frame)
    cv2.imshow('HSV Mask', color_mask)

    # Check for 'q' key to quit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
