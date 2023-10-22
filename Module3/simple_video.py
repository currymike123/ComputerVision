import cv2

def main():
    # Replace with your video file name
    video_file = "black_sphere.mp4"

    # Create a VideoCapture object
    video = cv2.VideoCapture(video_file)

    # Check if video opened successfully
    if not video.isOpened():
        print("Error: Could not open video.")
        return

    # Read and display video frames
    while True:
        ret, frame = video.read()

        # Break the loop if we failed to capture a frame
        if not ret:
            print("Can't receive frame or video is ended. Exiting ...")
            break

        # Display the resulting frame
        cv2.imshow('Video Playback', frame)

        # Press 'q' to exit the playback window
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
