import cv2
import numpy as np
import sys

# Video capture from file
cap = cv2.VideoCapture(0)

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

        # Setup SimpleBlobDetector parameters
    params = cv2.SimpleBlobDetector_Params()

    # Change Color
    params.filterByColor = False
    params.blobColor = 1

    # Change thresholds
    params.minThreshold = 0
    params.maxThreshold = 255

    # Filter by Area
    params.filterByArea = False
    params.minArea = 100
    params.maxArea = 5000

    # Filter by Circularity
    params.filterByCircularity = False
    params.minCircularity = 0.1

    # Filter by Convexity
    params.filterByConvexity = Falseq
    params.minConvexity = 0.87

    # Filter by Inertia
    params.filterByInertia = False
    params.minInertiaRatio = 0.01


    detector = cv2.SimpleBlobDetector_create(params)

    # Detect blobs
    keypoints = detector.detect(frame)

    # Draw detected blobs as red circles
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of the blob
    im_with_keypoints = cv2.drawKeypoints(frame, keypoints, None, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    

    # Display the original frame with the drawn contours
    cv2.imshow('Frame with tracked object', im_with_keypoints)

    # Check for 'q' key to quit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
