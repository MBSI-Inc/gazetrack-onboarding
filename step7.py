import cv2
import numpy as np
import mediapipe as mp
from sys import platform
from step6 import draw_eye_contour, write_direction_on_frame

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

def get_head_rotation(frame, landmarks):
    frame_h, frame_w, _ = frame.shape
    face_3d = []
    face_2d = []
    for idx, lm in enumerate(landmarks):
        if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
            if idx == 1:
                nose_2d = (lm.x * frame_w, lm.y * frame_h)
                nose_3d = (lm.x * frame_w, lm.y * frame_h, lm.z * 3000)
            x, y = int(lm.x * frame_w), int(lm.y * frame_h)
            # Get the 2D Coordinates
            face_2d.append([x, y])
            # Get the 3D Coordinates
            face_3d.append([x, y, lm.z])       
    # Convert it to the NumPy array
    face_2d = np.array(face_2d, dtype=np.float64)
    face_3d = np.array(face_3d, dtype=np.float64)
    # The camera matrix
    focal_length = 1 * frame_w
    cam_matrix = np.array([ [focal_length, 0, frame_h / 2],
                            [0, focal_length, frame_w / 2],
                            [0, 0, 1]])
    # The distortion parameters
    dist_matrix = np.zeros((4, 1), dtype=np.float64)
    # Solve PnP
    success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
    # Get rotational matrix
    rmat, jac = cv2.Rodrigues(rot_vec)
    # Get angles
    angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
    # Get the y rotation degree
    x = angles[0] * 360
    y = angles[1] * 360
    z = angles[2] * 360
    # Display the nose direction
    nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)
    p1 = (int(nose_2d[0]), int(nose_2d[1]))
    p2 = (int(nose_2d[0] + y * 10) , int(nose_2d[1] - x * 10))
    cv2.line(frame, p1, p2, (255, 0, 0), 3)
    # Add the text on the image
    cv2.putText(frame, "x: " + str(np.round(x,2)), (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, "y: " + str(np.round(y,2)), (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, "z: " + str(np.round(z,2)), (500, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return frame, y

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

            frame, head_rotation = get_head_rotation(frame, landmarks)

        # Flip the image horizontally for a selfie-view display.
        cv2.imshow("Step 5 Eye contour", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):  
            break

    cv2.destroyAllWindows()
    cam.release()

if __name__ == '__main__':
    main()