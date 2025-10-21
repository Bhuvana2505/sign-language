import os
import cv2
import numpy as np
import mediapipe as mp

# Paths
DATASET_DIR = "data"
OUTPUT_DIR = "keypoints"
SPLITS = ["train", "test"]

# Mediapipe setup
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Create output folders
for split in SPLITS:
    os.makedirs(os.path.join(OUTPUT_DIR, split), exist_ok=True)

def extract_keypoints_from_video(video_path):
    """Extract holistic (pose+hands+face) keypoints from a video."""
    cap = cv2.VideoCapture(video_path)
    holistic = mp_holistic.Holistic(static_image_mode=False)
    sequence = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(frame_rgb)

        # Each frame → a flattened vector of all keypoints
        keypoints = []
        if results.pose_landmarks:
            for lm in results.pose_landmarks.landmark:
                keypoints.extend([lm.x, lm.y, lm.z])
        else:
            keypoints.extend([0] * (33 * 3))

        if results.left_hand_landmarks:
            for lm in results.left_hand_landmarks.landmark:
                keypoints.extend([lm.x, lm.y, lm.z])
        else:
            keypoints.extend([0] * (21 * 3))

        if results.right_hand_landmarks:
            for lm in results.right_hand_landmarks.landmark:
                keypoints.extend([lm.x, lm.y, lm.z])
        else:
            keypoints.extend([0] * (21 * 3))

        sequence.append(keypoints)

    cap.release()
    holistic.close()
    return np.array(sequence)

for split in SPLITS:
    split_dir = os.path.join(DATASET_DIR, split)
    output_split_dir = os.path.join(OUTPUT_DIR, split)

    for label in os.listdir(split_dir):
        label_path = os.path.join(split_dir, label)
        if not os.path.isdir(label_path):
            continue

        output_label_dir = os.path.join(output_split_dir, label)
        os.makedirs(output_label_dir, exist_ok=True)

        for video_file in os.listdir(label_path):
            if not video_file.endswith((".mp4", ".mov")):
                continue

            video_path = os.path.join(label_path, video_file)
            output_file = os.path.join(output_label_dir, video_file.replace(".mp4", ".npy").replace(".mov", ".npy"))

            if os.path.exists(output_file):
                print(f"Skipping (exists): {output_file}")
                continue

            print(f"Processing {video_path} ...")
            keypoints = extract_keypoints_from_video(video_path)
            np.save(output_file, keypoints)

print("✅ Keypoint extraction complete. Saved in /keypoints folder.")
