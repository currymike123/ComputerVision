# Let's build background subtraction from scratch

import cv2
import numpy as np

def compute_difference(frame1, frame2):
    return cv2.absdiff(frame1, frame2)

cap = cv2.VideoCapture(0)

ret, reference_frame = cap.read()

# Parameters
num_frames_for_background = 100
count = 0

background_frame = None

while True:

    ret, frame = cap.read()

    if not ret:
        break

    if count < num_frames_for_background:

        float_frame = frame.astype(np.float32)

        if background_frame is None:
            background_frame = float_frame
        else:
            background_frame += float_frame

        count += 1

    elif count == num_frames_for_background:
        background = (background_frame / num_frames_for_background).astype(np.uint8)
    
    else:
        diff = compute_difference(background, frame)
        gray_dif = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, threshold_val = cv2.threshold(gray_diff, 100, 255, cv2.THRESH_BINARY)
        cv2.imshow("Motion", threshold_val)


    key = cv2.waitKey(1)
    if key == ord('Q') or key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


