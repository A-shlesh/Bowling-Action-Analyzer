import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions
import numpy as np
import time

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle = np.degrees(np.arccos(cosine_angle))

    return angle

options = vision.PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="backend/pose_landmarker.task"),
    running_mode=vision.RunningMode.VIDEO
)

cap = cv2.VideoCapture("backend/videos/bowling.mp4")

with vision.PoseLandmarker.create_from_options(options) as landmarker:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (800, 600))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp = int(time.time() * 1000)
        result = landmarker.detect_for_video(mp_image, timestamp)

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
            angle_left_leg = calculate_angle(hip, knee, ankle)
            cv2.putText(frame, str(int(angle_left_leg)), knee,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

            hip_r = points[24]
            knee_r = points[26]
            ankle_r = points[28]
            angle_right_leg = calculate_angle(hip_r, knee_r, ankle_r)
            cv2.putText(frame, str(int(angle_right_leg)), knee_r,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

            shoulder_l = points[11]
            elbow_l = points[13]
            wrist_l = points[15]
            angle_left_arm = calculate_angle(shoulder_l, elbow_l, wrist_l)
            cv2.putText(frame, str(int(angle_left_arm)), elbow_l,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

            shoulder_r = points[12]
            elbow_r = points[14]
            wrist_r = points[16]
            angle_right_arm = calculate_angle(shoulder_r, elbow_r, wrist_r)
            cv2.putText(frame, str(int(angle_right_arm)), elbow_r,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

            shoulder = points[11]
            hip_mid = points[23]
            dx = shoulder[0] - hip_mid[0]
            dy = shoulder[1] - hip_mid[1]
            spine_angle = np.degrees(np.arctan2(dy, dx))
            cv2.putText(frame, str(int(spine_angle)), hip_mid,
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

    timestamp = int(time.time() * 1000)

cap.release()
cv2.destroyAllWindows()