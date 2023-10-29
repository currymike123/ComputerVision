import cv2
import numpy as np

# Callback function for trackbars, necessary for creation of trackbars
def nothing(x):
    pass

# Video capture from webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Create a window for the sliders
cv2.namedWindow("Trackbars")

# Create trackbars for thresholding, area, circularity, convexity, and inertia
cv2.createTrackbar("Min Threshold", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("Max Threshold", "Trackbars", 255, 255, nothing)

# Create trackbar to set blob color on/off
cv2.createTrackbar("Blob Color On/Off", "Trackbars", 0, 1, nothing)
cv2.createTrackbar("Blob Color", "Trackbars", 1, 1, nothing)

cv2.createTrackbar("Min Area", "Trackbars", 100, 10000, nothing)
cv2.createTrackbar("Max Area", "Trackbars", 5000, 10000, nothing)

cv2.createTrackbar("Min Circularity", "Trackbars", 0, 100, nothing)
cv2.createTrackbar("Max Circularity", "Trackbars", 100, 100, nothing)

cv2.createTrackbar("Min Convexity", "Trackbars", 0, 100, nothing)
cv2.createTrackbar("Max Convexity", "Trackbars", 100, 100, nothing)

cv2.createTrackbar("Min Inertia Ratio", "Trackbars", 0, 100, nothing)
cv2.createTrackbar("Max Inertia Ratio", "Trackbars", 100, 100, nothing)

# Create trackbars for color thresholding
cv2.createTrackbar("Low-H", "Trackbars", 0, 179, nothing)
cv2.createTrackbar("High-H", "Trackbars", 179, 179, nothing)

cv2.createTrackbar("Low-S", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("High-S", "Trackbars", 255, 255, nothing)

cv2.createTrackbar("Low-V", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("High-V", "Trackbars", 255, 255, nothing)


while True:
    ret, frame = cap.read()

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    frame = cv2.resize(frame, None, fx = 0.4, fy = 0.4, interpolation=cv2.INTER_NEAREST)

    # Convert the captured frame to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Get current positions of all trackbars
    low_h = cv2.getTrackbarPos("Low-H", "Trackbars")
    high_h = cv2.getTrackbarPos("High-H", "Trackbars")
    low_s = cv2.getTrackbarPos("Low-S", "Trackbars")
    high_s = cv2.getTrackbarPos("High-S", "Trackbars")
    low_v = cv2.getTrackbarPos("Low-V", "Trackbars")
    high_v = cv2.getTrackbarPos("High-V", "Trackbars")

    # Create the mask using the current HSV range
    color_mask = cv2.inRange(hsv, (low_h, low_s, low_v), (high_h, high_s, high_v))

    # You can optionally apply some morphology operations to clean up the mask
    # E.g., cv2.erode() or cv2.dilate()

    # Apply the mask to extract the result and find blobs in this masked image
    result = cv2.bitwise_and(frame, frame, mask=color_mask)

    # Get current positions of all trackbars
    minThreshold = cv2.getTrackbarPos("Min Threshold", "Trackbars")
    maxThreshold = cv2.getTrackbarPos("Max Threshold", "Trackbars")
    blobColor = cv2.getTrackbarPos("Blob Color", "Trackbars")
    blobColorOnOff = cv2.getTrackbarPos("Blob Color On/Off", "Trackbars")
    minArea = max(1, cv2.getTrackbarPos("Min Area", "Trackbars"))  # Area cannot be 0
    maxArea = max(minArea, cv2.getTrackbarPos("Max Area", "Trackbars"))  # Max area must be >= minArea
    minCircularity = max(1, cv2.getTrackbarPos("Min Circularity", "Trackbars")) / 100  # Circularity ranges from 0.01 to 1, cannot be 0
    maxCircularity = max(minCircularity, cv2.getTrackbarPos("Max Circularity", "Trackbars")) / 100  # Max Circularity must be >= minCircularity
    minConvexity = max(1, cv2.getTrackbarPos("Min Convexity", "Trackbars")) / 100  # Same logic as Circularity
    maxConvexity = max(minConvexity, cv2.getTrackbarPos("Max Convexity", "Trackbars")) / 100  # Same logic as Circularity
    minInertiaRatio = max(1, cv2.getTrackbarPos("Min Inertia Ratio", "Trackbars")) / 100  # Same logic as Circularity
    maxInertiaRatio = max(minInertiaRatio, cv2.getTrackbarPos("Max Inertia Ratio", "Trackbars")) / 100  # Same logic as Circularity``

    # Setup SimpleBlobDetector parameters
    params = cv2.SimpleBlobDetector_Params()

    # Set thresholding values
    params.minThreshold = minThreshold
    params.maxThreshold = maxThreshold

    # Filter by Area
    params.filterByArea = True
    params.minArea = minArea
    params.maxArea = maxArea

    # Filter by Color
    params.filterByColor = blobColorOnOff
    params.blobColor = blobColor

    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = minCircularity

    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = minConvexity
    params.maxConvexity = maxConvexity

    # Filter by Inertia
    params.filterByInertia = True
    params.minInertiaRatio = minInertiaRatio
    params.maxInertiaRatio = maxInertiaRatio

    # Set up the detector with default parameters
    detector = cv2.SimpleBlobDetector_create(params)

    # Detect blobs
    keypoints = detector.detect(result)

    # Draw detected blobs as red circles
    im_with_keypoints = cv2.drawKeypoints(result, keypoints, np.array([]), (0, 0, 255),
                                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Show blobs
    cv2.imshow("Keypoints", im_with_keypoints)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture and destroy all windows
cap.release()
cv2.destroyAllWindows()
