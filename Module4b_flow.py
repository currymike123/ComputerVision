import cv2

# Load input video
cap = cv2.VideoCapture(0)

# Parameters for Shi-Tomasi corner detection
feature_params = dict(maxCorners=100, qualityLevel=.1, minDistance=7, blockSize=7)

# Parameters for Lucas-Kanade optical flow algorithm
lk_params = dict(winSize=(15,15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Initialize variables for tracking
old_gray = None
p0 = None

while True:
    # Read video frame
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect Shi-Tomasi corners in the frame
    if p0 is None:
        p0 = cv2.goodFeaturesToTrack(gray, mask=None, **feature_params)
        old_gray = gray.copy()
    
    # Apply Lucas-Kanade optical flow algorithm to track the corners
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, gray, p0, None, **lk_params)
    
    # Filter out the tracked points with poor quality
    good_new = p1[st==1]
    good_old = p0[st==1]
    
    # Draw the tracked points on the frame
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        frame = cv2.circle(frame, (int(a),int(b)), 5, (0,255,0), -1)
    
    # Update the variables for the next iteration
    old_gray = gray.copy()
    p0 = good_new.reshape(-1,1,2)
    
    # Display the tracked frame
    cv2.imshow('Face Tracking', frame)
    
    # Exit the loop if the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the display window
cap.release()
cv2.destroyAllWindows()
