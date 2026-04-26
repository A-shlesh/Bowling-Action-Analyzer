import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions
import numpy as np
import time
from collections import deque

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    return np.degrees(np.arccos(cosine_angle))

def get_point_3d(lm):
    return [lm.x, lm.y, lm.z]

options = vision.PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="backend/pose_landmarker.task"),
    running_mode=vision.RunningMode.VIDEO,
    output_segmentation_masks=False
)


def get_world_point(lm):
    return [lm.x, lm.y, lm.z]


cap = cv2.VideoCapture("backend/videos/bowling2.mp4")

smooth = {
    'left_leg':   deque(maxlen=5),
    'right_leg':  deque(maxlen=5),
    'left_arm':   deque(maxlen=5),
    'right_arm':  deque(maxlen=5),
    'spine':      deque(maxlen=5),
}

def smooth_angle(buf, new_val):
    buf.append(new_val)
    return sum(buf) / len(buf)

with vision.PoseLandmarker.create_from_options(options) as landmarker:


    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = int(time.time() * 1000)

        frame = cv2.resize(frame, (800, 600))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    
        result = landmarker.detect_for_video(mp_image, timestamp)

        if result.pose_landmarks and result.pose_world_landmarks:
            world = result.pose_world_landmarks[0]
        else:
            continue

        if result.pose_landmarks:
            h, w, _ = frame.shape
            points = []

            for lm in result.pose_landmarks[0]:
                cx, cy = int(lm.x * w), int(lm.y * h)
                points.append((cx, cy))
                cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

            connections = [
                (11,13),(13,15),(12,14),(14,16),
                (11,12),(11,23),(12,24),
                (23,25),(25,27),(24,26),(26,28),
                (23,24)
            ]

            for c in connections:
                cv2.line(frame, points[c[0]], points[c[1]], (255, 0, 0), 2)

            hip = points[23]
            knee = points[25]
            ankle = points[27]
            angle_left_leg = smooth_angle(smooth['left_leg'], calculate_angle(
            get_world_point(world[23]),
            get_world_point(world[25]),
            get_world_point(world[27])
            ))
            cv2.putText(frame, str(int(angle_left_leg)), knee,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

            hip_r = points[24]
            knee_r = points[26]
            ankle_r = points[28]
            angle_right_leg = smooth_angle(smooth['right_leg'], calculate_angle(
            get_world_point(world[24]),
            get_world_point(world[26]),
            get_world_point(world[28])
            ))
            cv2.putText(frame, str(int(angle_right_leg)), knee_r,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

            shoulder_l = points[11]
            elbow_l = points[13]
            wrist_l = points[15]
            angle_left_arm = smooth_angle(smooth['left_arm'], calculate_angle(
            get_world_point(world[11]),
            get_world_point(world[13]),
            get_world_point(world[15])
            ))

            cv2.putText(frame, str(int(angle_left_arm)), elbow_l,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

            shoulder_r = points[12]
            elbow_r = points[14]
            wrist_r = points[16]
            angle_right_arm = smooth_angle(smooth['right_arm'], calculate_angle(
            get_world_point(world[12]),
            get_world_point(world[14]),
            get_world_point(world[16])
            ))
            cv2.putText(frame, str(int(angle_right_arm)), elbow_r,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

            shoulder = points[11]
            hip_mid = points[23]
            dx = shoulder[0] - hip_mid[0]
            dy = shoulder[1] - hip_mid[1]
            angle_spine = smooth_angle(smooth['spine'], np.degrees(np.arctan2(
            world[11].y - world[23].y,
            world[11].x - world[23].x
            )))
            cv2.putText(frame, str(int(angle_spine)), hip_mid,
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,255), 2)

        cv2.imshow("Pose Detection", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
         break

        if key == ord('p'):
         while True:
          key2 = cv2.waitKey(0) & 0xFF
          if key2 == ord('p'):
            break
        

cap.release()
cv2.destroyAllWindows()