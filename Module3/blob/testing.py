import cv2
import numpy as np
import sys
print(sys.executable)

# Print the Python version
print("Python version")
print(sys.version)

# Print just the version number
print("Version info.")
print(sys.version_info)

# Video capture from file
cap = cv2.VideoCapture("shapes.mp4")

# Check if the video file opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # If the frame is not retrieved, then we've reached the end of the video
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Convert the frame to grayscale (required for Canny)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection; you might have to adjust the thresholds
    edges = cv2.Canny(gray, 100, 200)

    # This finds contours in the edged image. Contours are curves joining continuous points with the same color.
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw each contour only for visualisation purposes
    for contour in contours:
        # Here we are using boundingRect method to get the coordinates of the bounding box around the object (contour)
        x, y, w, h = cv2.boundingRect(contour)
        
        # Check if the size of the contour is large enough, considering noise and other smaller objects
        if cv2.contourArea(contour) > 25:  # this value (20) is arbitrary and depends on the object size in the video
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    

    # Display the original frame with the drawn contours
    cv2.imshow('Frame with tracked object', frame)

    # Check for 'q' key to quit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
