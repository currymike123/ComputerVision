import cv2

# Initialize the ORB detector
orb = cv2.ORB_create()

# Start capturing video
cap = cv2.VideoCapture(0)

while True:
    # Read a new frame
    ret, frame = cap.read()
    
    # Break the loop if no frame is returned
    if not ret:
        print("Unable to fetch frame. Exiting.")
        break
    
    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect keypoints
    keypoints, descriptors = orb.detectAndCompute(gray_frame, None)
    
    # Draw keypoints on the frame
    frame_with_keypoints = cv2.drawKeypoints(frame, keypoints, None, color=(0, 255, 0), flags=0)
    
    # Display the frame with keypoints
    cv2.imshow('Video with Keypoints', frame_with_keypoints)
    
    # Check for the exit command
    key = cv2.waitKey(1) 

    if key == ord('Q') or key == ord('q'):
        break
    

# Release the capture and close any open windows
cap.release()
cv2.destroyAllWindows()
