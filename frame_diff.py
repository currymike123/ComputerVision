import cv2
import numpy as np

def compute_difference(frame1, frame2):
    return cv2.absdiff(frame1, frame2)

cap = cv2.VideoCapture(0)

ret, reference_frame = cap.read()

while True:

    ret, frame = cap.read()

    if not ret:
        break

    diff = compute_difference(reference_frame, frame)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, threshold_val = cv2.threshold(gray_diff, 100, 255, cv2.THRESH_BINARY)
    cv2.imshow("Motion Detection", threshold_val)

    reference_frame = frame.copy()

    if cv2.waitKey(1)  & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
