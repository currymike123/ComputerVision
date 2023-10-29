import cv2
import numpy as np

# Start capturing video
video_capture = cv2.VideoCapture(0)

# Define the HSV color range for detection of the phone
lower_hsv = np.array([0, 0, 0])  # add the appropriate values
upper_hsv = np.array([60, 255, 255])  # add the appropriate values

# Set up blob detection parameters
params = cv2.SimpleBlobDetector_Params()
# Set parameters depending on your object and environment
# ... [your blob detection parameter settings here]

# Create a blob detector with the parameters
detector = cv2.SimpleBlobDetector_create(params)

while True:
    # Read a new frame
    ret, frame = video_capture.read()
    if not ret:
        break  # if failed to read a frame, exit the loop

    # Convert the frame to the HSV color space
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create a binary image, where white pixels (255) represent pixels that fall into the desired color range
    hsv_mask = cv2.inRange(hsv_frame, lower_hsv, upper_hsv)

    # Optional: apply some image processing techniques to refine the mask
    # ... [e.g., cv2.erode(), cv2.dilate()]

    # Use the mask to isolate the desired object (phone) in the frame
    object_isolated = cv2.bitwise_and(frame, frame, mask=hsv_mask)

    # Now use blob detection on the isolated object.
    keypoints = detector.detect(hsv_mask)  # Alternatively, you can use object_isolated depending on what works best

    # For visualization, draw the keypoints on the original image
    frame_with_keypoints = cv2.drawKeypoints(frame, keypoints, None, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Show the frame with detected object
    cv2.imshow('Detected Object', frame_with_keypoints)

    # Exit the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up and release the VideoCapture object
video_capture.release()
cv2.destroyAllWindows()
