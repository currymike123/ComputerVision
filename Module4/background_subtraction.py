import cv2
import numpy as np

def compute_difference(frame1, frame2):
    return cv2.absdiff(frame1, frame2)

cap = cv2.VideoCapture(0)

ret, reference_frame = cap.read()

# Parameters
num_frames_for_background = 100
count = 0

# Initialize a running sum of frames
background_accumulator = None

while True:

    ret, frame = cap.read()

    if not ret:
        break
    
    if count < num_frames_for_background:

        # Convert the frame to float
        float_frame = frame.astype(np.float32)

        # Add to background frame
        if background_accumulator is None:
            background_accumulator = float_frame
        else:
            background_accumulator += float_frame

        # Increment count
        count += 1
    elif count == num_frames_for_background:
        # Calculate the average to obtain the background frame
        background_frame = (background_accumulator / num_frames_for_background).astype(np.uint8)
        print("Background Frame")
        count += 1
    else:
        diff = compute_difference(background_frame, frame)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, threshold_val = cv2.threshold(gray_diff, 100, 255, cv2.THRESH_BINARY)
        cv2.imshow("Motion Detection", threshold_val)

    
    key = cv2.waitKey(1)
    if key == ord('Q') or key == ord('q'):
        break
    

cap.release()
cv2.destroyAllWindows()
