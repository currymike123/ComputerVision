import cv2
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, IntSlider

# Load the image outside the function so it's not repeatedly loaded
image = cv2.imread('blob_shapes.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB

def detect_blobs(lower_hue=0, upper_hue=10, lower_saturation=120, upper_saturation=255, lower_value=70, upper_value=255):
    # Convert the image to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Define the range of HSV values you're interested in
    lower_range = np.array([lower_hue, lower_saturation, lower_value])
    upper_range = np.array([upper_hue, upper_saturation, upper_value])
    
    # Create a mask for the HSV range
    mask = cv2.inRange(hsv, lower_range, upper_range)
    
    # Set up SimpleBlobDetector parameters
    blob_params = cv2.SimpleBlobDetector_Params()
    # Set your blob detection parameters here

    # Create a SimpleBlobDetector
    detector = cv2.SimpleBlobDetector_create(blob_params)
    
    # Use the mask to find the blobs and get the keypoints
    keypoints = detector.detect(mask)
    
    # Draw detected blobs as red circles on the original image
    image_with_keypoints = cv2.drawKeypoints(image, keypoints, np.array([]), (255,0,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    # Show the output image
    plt.figure(figsize=(10,10))
    plt.imshow(image_with_keypoints)
    plt.axis('off') # Hide axes
    plt.show()

# Create sliders for the parameters
interact(detect_blobs,
         lower_hue=IntSlider(min=0, max=180, step=1, value=0, continuous_update=False),
         upper_hue=IntSlider(min=0, max=180, step=1, value=10, continuous_update=False),
         lower_saturation=IntSlider(min=0, max=255, step=1, value=120, continuous_update=False),
         upper_saturation=IntSlider(min=0, max=255, step=1, value=255, continuous_update=False),
         lower_value=IntSlider(min=0, max=255, step=1, value=70, continuous_update=False),
         upper_value=IntSlider(min=0, max=255, step=1, value=255, continuous_update=False))
