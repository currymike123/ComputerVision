import cv2
import numpy as np

# Initialize video capture from default camera
cap = cv2.VideoCapture(0)

# Initialize Shi-Tomasi parameters
feature_params = dict(qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)

# Take first frame and detect corners in it
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, maxCorners=100, mask=None, **feature_params)

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)

while True:
    ret, frame = cap.read()

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate optical flow for each corner
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **feature_params)

    # Select good points
    good_new = p1[st == 1]
    good_old = p0[st == 1]

    # Find transformation matrix
    M, _ = cv2.findHomography(good_old, good_new, cv2.RANSAC)

    # Apply transformation to bounding box
    h, w = old_frame.shape[:2]
    bbox = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    new_bbox = cv2.perspectiveTransform(bbox, M)

    # Draw bounding box on frame
    frame = cv2.polylines(frame, [np.int32(new_bbox)], True, (0, 255, 0), 3, cv2.LINE_AA)

    # Show result
    cv2.imshow('frame', frame)

    # Update previous frame and points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

    # Exit on ESC
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
