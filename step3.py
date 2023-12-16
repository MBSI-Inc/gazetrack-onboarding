import cv2 # Load OpenCV module
from sys import platform

# Setting for the camera output resolution. Change this if the window
# open up too small or too large.
CAM_WIDTH = 1280
CAM_HEIGHT = 720

def main():
    # Some setting that make OpenCV startup faster
    if platform == "win32":
        cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    else:
        cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    cam.set(cv2.CAP_PROP_FPS, 30)
    cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

    # This loop will run forever
    while True:
        # Fetch a frame image from camera
        success, frame = cam.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
        
        # Mirror the image
        frame = cv2.flip(frame, 1)
        # Show the frame on a new window called "Camera"
        cv2.imshow("Step 3 Camera", frame)

        # Break the loop if user press Q
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Make sure OpenCV quit gracefully
    cv2.destroyAllWindows()
    cam.release()

if __name__ == '__main__':
    main()