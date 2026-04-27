import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions
import numpy as np
import time
import os
import hashlib
from collections import deque

# ── Config ────────────────────────────────────────────────────────────────────
VIDEO_PATH  = "backend/videos/bowling2.mp4"
MODEL_PATH  = "backend/pose_landmarker.task"
OUTPUT_DIR  = "backend/output"

# ── Helpers ───────────────────────────────────────────────────────────────────
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

# ── Setup ─────────────────────────────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)

output_path = get_output_path(VIDEO_PATH)
if os.path.exists(output_path):
    print(f"Already processed: {output_path}")
    exit()

options = vision.PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=vision.RunningMode.VIDEO,
    output_segmentation_masks=False,
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

print(f"Processing: {VIDEO_PATH}")
print(f"Output    : {output_path}")

# ── Main loop ─────────────────────────────────────────────────────────────────
with vision.PoseLandmarker.create_from_options(options) as landmarker:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = int(time.time() * 1000)

        small     = cv2.resize(frame, (480, 360))
        rgb       = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result    = landmarker.detect_for_video(mp_image, timestamp)

        frame = cv2.resize(frame, (orig_w, orig_h))

        if result.pose_landmarks and result.pose_world_landmarks:
            world  = result.pose_world_landmarks[0]
            h, w, _ = frame.shape
            points = []

            for lm in result.pose_landmarks[0]:
                cx, cy = int(lm.x * w), int(lm.y * h)
                points.append((cx, cy))
                cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

            for c in connections:
                cv2.line(frame, points[c[0]], points[c[1]], (255, 0, 0), 2)

            angle_left_leg = smooth_angle(smooth['left_leg'], calculate_angle(
                get_world_point(world[23]),
                get_world_point(world[25]),
                get_world_point(world[27])
            ))
            cv2.putText(frame, str(int(angle_left_leg)), points[25],
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

            angle_right_leg = smooth_angle(smooth['right_leg'], calculate_angle(
                get_world_point(world[24]),
                get_world_point(world[26]),
                get_world_point(world[28])
            ))
            cv2.putText(frame, str(int(angle_right_leg)), points[26],
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

            angle_left_arm = smooth_angle(smooth['left_arm'], calculate_angle(
                get_world_point(world[11]),
                get_world_point(world[13]),
                get_world_point(world[15])
            ))
            cv2.putText(frame, str(int(angle_left_arm)), points[13],
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

            angle_right_arm = smooth_angle(smooth['right_arm'], calculate_angle(
                get_world_point(world[12]),
                get_world_point(world[14]),
                get_world_point(world[16])
            ))
            cv2.putText(frame, str(int(angle_right_arm)), points[14],
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

            angle_spine = smooth_angle(smooth['spine'], np.degrees(np.arctan2(
                world[11].y - world[23].y,
                world[11].x - world[23].x
            )))
            cv2.putText(frame, str(int(angle_spine)), points[23],
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,255), 2)

        writer.write(frame)
        cv2.imshow("Pose Detection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('p'):
            while True:
                if cv2.waitKey(0) & 0xFF == ord('p'):
                    break

cap.release()
writer.release()
cv2.destroyAllWindows()
print(f"Saved: {output_path}")