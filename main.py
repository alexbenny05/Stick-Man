import cv2
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

# --- INIT POSE LANDMARKER ---
base_options = python.BaseOptions(model_asset_path="pose_landmarker_full.task")
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_poses=1
)
detector = vision.PoseLandmarker.create_from_options(options)

# --- START CAMERA ---
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    timestamp_ms = int(time.time() * 1000)
    result = detector.detect_for_video(mp_image, timestamp_ms)

    # Create a blank background for stick man
    h, w, _ = frame.shape
    stickman = np.zeros((h, w, 3), dtype=np.uint8)

    if result.pose_landmarks:
        landmarks = result.pose_landmarks[0]  # first person

        def get_point(idx):
            return int(landmarks[idx].x * w), int(landmarks[idx].y * h)

        # Key joints
        head = get_point(0)   # Nose
        left_shoulder = get_point(11)
        right_shoulder = get_point(12)
        left_elbow = get_point(13)
        right_elbow = get_point(14)
        left_wrist = get_point(15)
        right_wrist = get_point(16)
        left_hip = get_point(23)
        right_hip = get_point(24)
        left_knee = get_point(25)
        right_knee = get_point(26)
        left_ankle = get_point(27)
        right_ankle = get_point(28)

        # Draw stick man with thick colored lines
        cv2.circle(stickman, head, 40, (0, 255, 0), -1)  # Head (larger green filled circle)

        # Torso (yellow)
        cv2.line(stickman, left_shoulder, right_shoulder, (0, 255, 255), 6)
        cv2.line(stickman, left_shoulder, left_hip, (0, 255, 255), 6)
        cv2.line(stickman, right_shoulder, right_hip, (0, 255, 255), 6)
        cv2.line(stickman, left_hip, right_hip, (0, 255, 255), 6)

        # Arms (blue)
        cv2.line(stickman, left_shoulder, left_elbow, (255, 0, 0), 6)
        cv2.line(stickman, left_elbow, left_wrist, (255, 0, 0), 6)
        cv2.line(stickman, right_shoulder, right_elbow, (255, 0, 0), 6)
        cv2.line(stickman, right_elbow, right_wrist, (255, 0, 0), 6)

        # Legs (red)
        cv2.line(stickman, left_hip, left_knee, (0, 0, 255), 6)
        cv2.line(stickman, left_knee, left_ankle, (0, 0, 255), 6)
        cv2.line(stickman, right_hip, right_knee, (0, 0, 255), 6)
        cv2.line(stickman, right_knee, right_ankle, (0, 0, 255), 6)

    # Show stick man in its own window
    cv2.imshow("Stick Man Figure", stickman)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()