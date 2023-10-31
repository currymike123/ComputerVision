import cv2
import numpy as np

def compute_difference(frame1, frame2):
    """
    Compute the absolute difference between two frames.
    """
    assert frame1.shape == frame2.shape, "Frames have different shapes."
    diff = np.abs(frame1.astype(int) - frame2.astype(int)).astype(np.uint8)
    return diff

# Dummy callback function for trackbars
def nothing(x):
    pass

# Create named window for adding trackbars
cv2.namedWindow("Motion Mask Custom")

# Add trackbars for min and max area
cv2.createTrackbar('Min Area', 'Motion Mask Custom', 0, 5000, nothing)
cv2.createTrackbar('Max Area', 'Motion Mask Custom', 500, 5000, nothing)

cap = cv2.VideoCapture(0)
ret, reference_frame = cap.read()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    diff_custom = compute_difference(reference_frame, frame)
    gray_diff_custom = cv2.cvtColor(diff_custom, cv2.COLOR_BGR2GRAY)
    _, thresholded = cv2.threshold(gray_diff_custom, 25, 255, cv2.THRESH_BINARY)

    # Retrieve min and max area values from trackbars
    min_area = cv2.getTrackbarPos('Min Area', 'Motion Mask Custom')
    max_area = cv2.getTrackbarPos('Max Area', 'Motion Mask Custom')

    contours_custom, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours_custom:
        area = cv2.contourArea(contour)
        if area >= min_area and area <= max_area:
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
    cv2.imshow("Motion Mask Custom", frame)
    
    reference_frame = frame.copy()

    # Exit loop on 'q' key press
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
