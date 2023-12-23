import cv2
import numpy as np
import mediapipe as mp
from sys import platform
from step5 import get_numpy_points_from_landmarks

CAM_WIDTH = 1280
CAM_HEIGHT = 720
MAX_PEOPLE = 1
EMA_ALPHA = 0.3

RIGHT_UPPER_EYELID_LANDMARKS = [160, 161, 157, 158, 159]
RIGHT_LOWER_EYELID_LANDMARKS = [163, 144, 145, 153, 154]
RIGHT_EYE_LANDMARKS = RIGHT_UPPER_EYELID_LANDMARKS + RIGHT_LOWER_EYELID_LANDMARKS
RIGHT_IRIS_LANDMARKS = [472, 469, 470, 471]

LEFT_UPPER_EYELID_LANDMARKS = [384, 385, 386, 387, 388]
LEFT_LOWER_EYELID_LANDMARKS = [390, 373, 374, 380, 381]
LEFT_EYE_LANDMARKS = LEFT_UPPER_EYELID_LANDMARKS + LEFT_LOWER_EYELID_LANDMARKS
LEFT_IRIS_LANDMARKS = [474, 475, 476, 477]

def determine_direction(pupil_x, eye_x_low, eye_x_high):
    direction_ratio = (pupil_x - eye_x_low) / (eye_x_high - eye_x_low)
    # Convert from [0, 1] range into [-1, 1] range so it's easier to use
    direction_ratio = (2 * direction_ratio) - 1
    return direction_ratio


def write_direction_on_frame(direction, frame):
    if direction < 0:
        cv2.putText(frame, "left", (300, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA,)
    elif direction > 0:
        cv2.putText(frame, "right", (300, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
    return frame
    
def draw_eye_contour(frame, landmarks, right=False):
        # `right` = is this the right eye (true) or left eye (false).
        # returns `frame`, which is the face cam image with eyes and iris box annotated on top.
        if (right):
            eye_landmarks = RIGHT_EYE_LANDMARKS
            iris_landmarks = RIGHT_IRIS_LANDMARKS
        else:
            eye_landmarks = LEFT_EYE_LANDMARKS
            iris_landmarks = LEFT_IRIS_LANDMARKS

        # You can change these toggle variable to see more or less details
        show_eye_bounding_box = True
        show_eye_landmark = True
        show_iris_bounding_box = True
        show_iris_landmark = False

        eye_pts = get_numpy_points_from_landmarks(frame, landmarks, eye_landmarks)
        iris_pts = get_numpy_points_from_landmarks(frame, landmarks, iris_landmarks)

        if show_eye_bounding_box:
            # Draw rectangle around eye with green color
            # (x, y) is the location of the bounding rect, (w, h) is the size.
            x, y, w, h = cv2.boundingRect(eye_pts)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 1)

        if show_eye_landmark:
            # Draw all eye landmarks as white dot with green line
            hull_indices = cv2.convexHull(eye_pts, returnPoints=False)
            for point in eye_pts:
                cv2.circle(frame, tuple(point), 2, (255, 255, 255), -1)
            hull_points = [eye_pts[index] for index in hull_indices]
            cv2.drawContours(frame, [np.array(hull_points)], 0, (0, 255, 0), 1)

        if show_iris_bounding_box:
            # Draw circle around iris with red color
            center, radius = cv2.minEnclosingCircle(iris_pts)
            center = tuple(map(int, center))
            radius = int(radius/2)
            cv2.circle(frame, center, radius, (0, 0, 255), 1)

        if show_iris_landmark:
            # Draw all iris landmarks as white dot with red line
            hull_indices = cv2.convexHull(iris_pts, returnPoints=False)
            for point in iris_pts:
                cv2.circle(frame, tuple(point), 2, (255, 255, 255), -1)
            hull_points = [iris_pts[index] for index in hull_indices]
            cv2.drawContours(frame, [np.array(hull_points)], 0, (0, 0, 255), 1)

        eye_x_low = x
        eye_x_high = x+w
        direction_ratio = determine_direction(center[0], eye_x_low, eye_x_high)
        return frame, direction_ratio

def main():
    if platform == "win32":
        cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    else:
        cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    cam.set(cv2.CAP_PROP_FPS, 30)
    cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

    # This is for smoothing
    ema_left = 0
    ema_right = 0

    # Setup facemesh setting
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=MAX_PEOPLE, refine_landmarks=True, 
        min_detection_confidence=0.5, min_tracking_confidence=0.5)

    while True:
        success, frame = cam.read()
        frame = cv2.flip(frame, 1)
        if not success:
            print("Ignoring empty camera frame.")
            continue

        output = face_mesh.process(frame)
        landmark_points = output.multi_face_landmarks
        if landmark_points:
            landmarks = landmark_points[0].landmark
            # Left eye
            frame, left_direction_ratio = draw_eye_contour(frame, landmarks, False)
            ema_left = (EMA_ALPHA * left_direction_ratio) + ((1 - EMA_ALPHA) * ema_left)
            left_direction_ratio = ema_left
            
            # Right eye
            frame, right_direction_ratio = draw_eye_contour(frame, landmarks, True)
            ema_right = (EMA_ALPHA * right_direction_ratio) + ((1 - EMA_ALPHA) * ema_right)
            right_direction_ratio = ema_right

            # Draw the eye direction ratio on frame
            cv2.putText(frame, f"{round(right_direction_ratio, 2)}", (300, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA,)
            cv2.putText(frame, f"{round(left_direction_ratio, 2)}", (100, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA,)


            gaze_direction = (left_direction_ratio + right_direction_ratio) / 2

            cv2.putText(frame, f"{round(gaze_direction, 2)}", (100, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA,)

            frame = write_direction_on_frame(gaze_direction, frame)

        # Flip the image horizontally for a selfie-view display.
        cv2.imshow("Step 5 Eye contour", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):  
            break

    cv2.destroyAllWindows()
    cam.release()

if __name__ == '__main__':
    main()