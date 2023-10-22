import cv2
import numpy as np

def is_head_like(contour, min_size=500):
    """
    Determines if a contour is head-like based on its properties.

    :param contour: The contour to check.
    :param min_size: The minimum size below which a contour is not considered.
    :return: True if the contour is head-like, False otherwise.
    """
    # Calculate area and perimeter
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)

    # A head is unlikely to be smaller than a certain size
    if area < min_size:
        return False

    # Check if the contour is roughly circular
    circularity = 4 * np.pi * area / (perimeter * perimeter)
    if 0.5 < circularity < 5:  # these values can be adjusted
        return True

    return False

# Main script starts here
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if is_head_like(contour):
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow('Head Tracking', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
