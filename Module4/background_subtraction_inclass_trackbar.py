import cv2
import numpy as np

def compute_difference(frame1, frame2):
    return cv2.absdiff(frame1, frame2)

def nothing(x):
    pass

cap = cv2.VideoCapture(0)

# Create a window for the trackbars
cv2.namedWindow('Settings')

# Create trackbars for erosion and dilation iterations
cv2.createTrackbar('Erosion Iterations', 'Settings', 0, 10, nothing)
cv2.createTrackbar('Dilation Iterations', 'Settings', 0, 10, nothing)
cv2.createTrackbar('Area', 'Settings', 1000, 10000, nothing)

# Parameters
num_frames_for_background = 100
count = 0

background_frame = None

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    # Get current positions of the trackbars
    erosion_iter = cv2.getTrackbarPos('Erosion Iterations', 'Settings')
    dilation_iter = cv2.getTrackbarPos('Dilation Iterations', 'Settings')
    min_area = cv2.getTrackbarPos('Area', 'Settings')

    if count < num_frames_for_background:
        # Build the background model
        float_frame = frame.astype(np.float32)
        if background_frame is None:
            background_frame = float_frame
        else:
            background_frame += float_frame
        count += 1

    elif count == num_frames_for_background:
        # Calculate the average to obtain the background frame
        background = (background_frame / num_frames_for_background).astype(np.uint8)
        count += 1  # Ensure this block does not run again

    else:
        # Compute the absolute difference between the background and the current frame
        diff = compute_difference(background, frame)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, threshold_val = cv2.threshold(gray_diff, 100, 255, cv2.THRESH_BINARY)

        # Apply erosion and dilation
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        threshold_val = cv2.erode(threshold_val, kernel, iterations=erosion_iter)
        threshold_val = cv2.dilate(threshold_val, kernel, iterations=dilation_iter)

        # Find contours
        contours, _ = cv2.findContours(threshold_val.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > min_area:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.imshow("Motion", threshold_val)
    cv2.imshow("Live Feed", frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
