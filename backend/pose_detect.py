import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions
import numpy as np
import time
import os
import json
import hashlib
from collections import deque

VIDEO_PATH  = "backend/videos/bowling2.mp4"
MODEL_PATH  = "backend/pose_landmarker.task"
OUTPUT_DIR  = "backend/output"

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

def get_world_point(lm):
    return [lm.x, lm.y, lm.z]

def smooth_angle(buf, val):
    buf.append(val)
    return sum(buf) / len(buf)

def get_output_path(video_path):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    with open(video_path, "rb") as f:
        file_hash = hashlib.md5(f.read(1024 * 1024)).hexdigest()[:8]
    return os.path.join(OUTPUT_DIR, f"{video_name}_{file_hash}.mp4")

def get_joint_color(visibility):
    if visibility > 0.8:
        return (0, 255, 0)
    elif visibility > 0.5:
        return (0, 165, 255)
    else:
        return (0, 0, 255)

# Setup
os.makedirs(OUTPUT_DIR, exist_ok=True)

output_path = get_output_path(VIDEO_PATH)
json_path   = output_path.replace(".mp4", "_angles.json")

if os.path.exists(output_path):
    print(f"Already processed: {output_path}")
    exit()

options = vision.PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=vision.RunningMode.VIDEO,
    min_pose_detection_confidence=0.6,
    min_pose_presence_confidence=0.6,
    min_tracking_confidence=0.6,
)

smooth = {
    'left_leg':  deque(maxlen=10),
    'right_leg': deque(maxlen=10),
    'left_arm':  deque(maxlen=10),
    'right_arm': deque(maxlen=10),
    'spine':     deque(maxlen=10),
}

connections = [
    (11,13),(13,15),(12,14),(14,16),
    (11,12),(11,23),(12,24),
    (23,25),(25,27),(24,26),(26,28),
    (23,24)
]

cap    = cv2.VideoCapture(VIDEO_PATH)
fps    = cap.get(cv2.CAP_PROP_FPS)
orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

writer = cv2.VideoWriter(
    output_path,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (orig_w, orig_h)
)

frame_data = []
frame_count = 0
prev_time   = time.time()

print(f"Processing : {VIDEO_PATH}")

with vision.PoseLandmarker.create_from_options(options) as landmarker:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = int(time.time() * 1000)

        small = cv2.resize(frame, (480, 360))
        rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = landmarker.detect_for_video(mp_image, timestamp)

        frame = cv2.resize(frame, (orig_w, orig_h))

        curr_time = time.time()
        fps_display = int(1 / (curr_time - prev_time + 1e-6))
        prev_time = curr_time

        cv2.putText(frame, f"FPS: {fps_display}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if result.pose_landmarks and result.pose_world_landmarks:
            world = result.pose_world_landmarks[0]
            h, w, _ = frame.shape
            points = []

            for lm in result.pose_landmarks[0]:
                cx, cy = int(lm.x * w), int(lm.y * h)
                points.append((cx, cy))
                cv2.circle(frame, (cx, cy), 4, get_joint_color(lm.visibility), -1)

            for c in connections:
                cv2.line(frame, points[c[0]], points[c[1]], (255, 0, 0), 2)

            angle_left_leg = smooth_angle(smooth['left_leg'], calculate_angle(
                get_world_point(world[23]), get_world_point(world[25]), get_world_point(world[27])
            ))

            angle_right_leg = smooth_angle(smooth['right_leg'], calculate_angle(
                get_world_point(world[24]), get_world_point(world[26]), get_world_point(world[28])
            ))

            angle_left_arm = smooth_angle(smooth['left_arm'], calculate_angle(
                get_world_point(world[11]), get_world_point(world[13]), get_world_point(world[15])
            ))

            angle_right_arm = smooth_angle(smooth['right_arm'], calculate_angle(
                get_world_point(world[12]), get_world_point(world[14]), get_world_point(world[16])
            ))

            angle_spine = smooth_angle(smooth['spine'], np.degrees(np.arctan2(
                world[11].y - world[23].y,
                world[11].x - world[23].x
            )))

            cv2.putText(frame, str(int(angle_left_leg)), points[25], cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
            cv2.putText(frame, str(int(angle_right_leg)), points[26], cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
            cv2.putText(frame, str(int(angle_left_arm)), points[13], cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
            cv2.putText(frame, str(int(angle_right_arm)), points[14], cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
            cv2.putText(frame, str(int(angle_spine)), points[23], cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,255), 2)

            frame_data.append({
                "frame": frame_count,
                "left_leg": int(angle_left_leg),
                "right_leg": int(angle_right_leg),
                "left_arm": int(angle_left_arm),
                "right_arm": int(angle_right_arm),
                "spine": int(angle_spine)
            })

        writer.write(frame)
        cv2.imshow("Pose Detection", frame)
        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# 🔥 SUMMARY FEATURE
avg_left_leg  = int(np.mean([f["left_leg"] for f in frame_data])) if frame_data else 0
avg_right_leg = int(np.mean([f["right_leg"] for f in frame_data])) if frame_data else 0
avg_left_arm  = int(np.mean([f["left_arm"] for f in frame_data])) if frame_data else 0
avg_right_arm = int(np.mean([f["right_arm"] for f in frame_data])) if frame_data else 0
avg_spine     = int(np.mean([f["spine"] for f in frame_data])) if frame_data else 0

with open(json_path, "w") as f:
    json.dump({
        "video": VIDEO_PATH,
        "summary": {
            "avg_left_leg": avg_left_leg,
            "avg_right_leg": avg_right_leg,
            "avg_left_arm": avg_left_arm,
            "avg_right_arm": avg_right_arm,
            "avg_spine": avg_spine
        },
        "frames": frame_data
    }, f, indent=2)

cap.release()
writer.release()
cv2.destroyAllWindows()

print("Saved video:", output_path)
print("Saved JSON :", json_path)