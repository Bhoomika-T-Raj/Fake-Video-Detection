import os
import cv2
import numpy as np
from tqdm import tqdm

# Dataset paths
REAL_PATH = "Dataset/real"
FAKE_PATH = "Dataset/fake"
OUTPUT_DIR = "./processed_frames"  # Directory to save processed frames
OUTPUT_FRAME_SIZE = (128, 128)  # Frame dimensions
FRAME_COUNT = 10  # Number of frames to extract per video

# Ensure output directories exist
REAL_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "real")
FAKE_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "fake")
os.makedirs(REAL_OUTPUT_PATH, exist_ok=True)
os.makedirs(FAKE_OUTPUT_PATH, exist_ok=True)

# Function to extract and save frames for a video
def save_frames(video_path, output_dir, output_size=(128, 128), frame_count=10):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(total_frames // frame_count, 1)  # Uniform sampling

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_output_dir = os.path.join(output_dir, video_name)
    os.makedirs(video_output_dir, exist_ok=True)

    frame_saved = 0
    for i in range(frame_count):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, output_size)
        frame_file = os.path.join(video_output_dir, f"frame_{i:03d}.npy")  # Save as .npy file
        np.save(frame_file, frame)
        frame_saved += 1

    cap.release()
    return frame_saved

# Process real videos
print("Processing and saving real video frames...")
for video_file in tqdm(os.listdir(REAL_PATH)):
    video_path = os.path.join(REAL_PATH, video_file)
    save_frames(video_path, REAL_OUTPUT_PATH, output_size=OUTPUT_FRAME_SIZE, frame_count=FRAME_COUNT)

# Process fake videos
print("Processing and saving fake video frames...")
for video_file in tqdm(os.listdir(FAKE_PATH)):
    video_path = os.path.join(FAKE_PATH, video_file)
    save_frames(video_path, FAKE_OUTPUT_PATH, output_size=OUTPUT_FRAME_SIZE, frame_count=FRAME_COUNT)

print(f"Frames have been saved in '{OUTPUT_DIR}'.")
