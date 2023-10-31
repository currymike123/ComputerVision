import cv2
import numpy as np

def nothing(x):
    pass

# Initialize the video capture object
cap = cv2.VideoCapture("shapes.mp4")
cap = cv2.VideoCapture(0)



# Create a window for the trackbars
cv2.namedWindow("Trackbars")

# Create trackbars for color change (HSV Lower and Upper bounds for 'red' here)
cv2.createTrackbar("Low H", "Trackbars", 0, 179, nothing)
cv2.createTrackbar("High H", "Trackbars", 179, 179, nothing)
cv2.createTrackbar("Low S", "Trackbars", 120, 255, nothing)
cv2.createTrackbar("High S", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("Low V", "Trackbars", 120, 255, nothing)
cv2.createTrackbar("High V", "Trackbars", 255, 255, nothing)

# Create trackbars for adjusting the minimum and maximum area
cv2.createTrackbar("Min Area", "Trackbars", 400, 2000, nothing)
cv2.createTrackbar("Max Area", "Trackbars", 2000, 10000, nothing) # Adjust the max limit as needed

while True:
    ret, frame = cap.read()

    # Loop the video
    # Check if the video was opened successfully
    if not ret:
        print("Reached the end of the video, looping...")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    # Resize the frame    
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
    
    # Convert the frame to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Get the new values of the trackbar in real-time
    low_h = cv2.getTrackbarPos("Low H", "Trackbars")
    high_h = cv2.getTrackbarPos("High H", "Trackbars")
    low_s = cv2.getTrackbarPos("Low S", "Trackbars")
    high_s = cv2.getTrackbarPos("High S", "Trackbars")
    low_v = cv2.getTrackbarPos("Low V", "Trackbars")
    high_v = cv2.getTrackbarPos("High V", "Trackbars")
    min_area = cv2.getTrackbarPos("Min Area", "Trackbars")
    max_area = cv2.getTrackbarPos("Max Area", "Trackbars")

    # Set the lower and upper HSV values to detect colors within the range
    lower_red = np.array([low_h, low_s, low_v])
    upper_red = np.array([high_h, high_s, high_v])

    # Create a mask that matches the colors within the specified boundaries
    mask = cv2.inRange(hsv, lower_red, upper_red)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Draw bounding boxes around the detected objects that meet the area criteria
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area < area < max_area:  # filtering by area
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Show the frames
    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)

    # Wait for the 'q' key to stop the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
