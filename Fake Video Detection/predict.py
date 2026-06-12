import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import Xception
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import TimeDistributed, Flatten
# #
# # # Configuration
# # MODEL_PATH = "deepfake_detection_model.keras"
# # FRAME_COUNT = 10
# # FRAME_SIZE = (128, 128)
# #
# # # Load the trained model
# # print("Loading model...")
# # model = load_model(MODEL_PATH)
# # print("Model loaded successfully.")
# #
# # # Function to extract frames from video
# # def extract_frames(video_path, output_size=(128, 128), frame_count=10):
# #     cap = cv2.VideoCapture(video_path)
# #     frames = []
# #     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# #     step = max(total_frames // frame_count, 1)  # Uniform sampling
# #
# #     for i in range(frame_count):
# #         cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
# #         ret, frame = cap.read()
# #         if not ret:
# #             break
# #         frame = cv2.resize(frame, output_size)
# #         frames.append(frame)
# #     cap.release()
# #     return np.array(frames)
# #
# # # Function to preprocess frames
# # def preprocess_frames(frames):
# #     frames = frames / 255.0  # Normalize
# #     return np.expand_dims(frames, axis=0)  # Add batch dimension
# #
# # # Predict function
# # def predict_video_class(video_path):
# #     print(f"Processing video: {video_path}")
# #     frames = extract_frames(video_path, output_size=FRAME_SIZE, frame_count=FRAME_COUNT)
# #
# #     if len(frames) < FRAME_COUNT:
# #         print("Error: Video does not have enough frames for prediction.")
# #         return
# #
# #     frames = preprocess_frames(frames)
# #
# #     predictions = model.predict(frames)
# #     predicted_class = np.argmax(predictions, axis=1)[0]
# #     class_labels = {0: "Real", 1: "Fake"}
# #
# #     print(f"Prediction: {class_labels[predicted_class]} (Confidence: {predictions[0][predicted_class]:.2f})")
# #
# #     return class_labels[predicted_class]
# #
# # if __name__ == "__main__":
# #     video_file = 'Dataset/real/01__outside_talking_still_laughing.mp4'
# #
# #     if not os.path.isfile(video_file):
# #         print("Error: File not found.")
# #     else:
# #         result = predict_video_class(video_file)
#
import os
import cv2
import numpy as np
import tensorflow as tf
import shutil
from tensorflow.keras.models import load_model

# Configuration
MODEL_PATH = "deepfake_detection_model.keras"
DATASET_PATH = "dataset"  # Folder containing 'real' and 'fake' directories
TESTING_VIDEOS_PATH = "testing_videos"
FRAME_COUNT = 10
FRAME_SIZE = (128, 128)
TOP_N_VIDEOS = 15


# Load the trained model
def load_trained_model():
    return load_model(MODEL_PATH)


model = load_trained_model()


# Function to extract frames from a video
def extract_frames(video_path, output_size=(128, 128), frame_count=10):
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(total_frames // frame_count, 1)

    for i in range(frame_count):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, output_size)
        frames.append(frame)
    cap.release()
    return np.array(frames)


# Function to preprocess frames
def preprocess_frames(frames):
    frames = frames / 255.0
    return np.expand_dims(frames, axis=0)


# Predict function
def predict_video_class(video_path):
    frames = extract_frames(video_path, output_size=FRAME_SIZE, frame_count=FRAME_COUNT)
    if len(frames) < FRAME_COUNT:
        return None, None  # Not enough frames for prediction
    frames = preprocess_frames(frames)
    predictions = model.predict(frames)
    predicted_class = np.argmax(predictions, axis=1)[0]
    class_labels = {0: "fake", 1: "real"}
    confidence = predictions[0][predicted_class]
    return class_labels[predicted_class], confidence


# Process dataset and move correctly classified videos
def process_dataset():
    if not os.path.exists(TESTING_VIDEOS_PATH):
        os.makedirs(TESTING_VIDEOS_PATH)

    video_files_real = []
    video_files_fake = []

    # Get list of videos from dataset (real & fake directories)
    for class_name in ["real", "fake"]:
        class_path = os.path.join(DATASET_PATH, class_name)
        if os.path.exists(class_path):
            videos = [(os.path.join(class_path, vid), class_name) for vid in os.listdir(class_path) if
                      vid.endswith((".mp4", ".avi", ".mov"))]
            if class_name == "real":
                video_files_real.extend(videos)
            else:
                video_files_fake.extend(videos)

    # Process only the top N videos for each class
    video_files_real = video_files_real[:TOP_N_VIDEOS]
    video_files_fake = video_files_fake[:TOP_N_VIDEOS]

    for video_files in [video_files_real, video_files_fake]:
        for video_path, actual_class in video_files:
            predicted_class, confidence = predict_video_class(video_path)
            if predicted_class and predicted_class == actual_class:
                correct_class_path = os.path.join(TESTING_VIDEOS_PATH, predicted_class)
                if not os.path.exists(correct_class_path):
                    os.makedirs(correct_class_path)

                # Move correctly classified video
                shutil.move(video_path, os.path.join(correct_class_path, os.path.basename(video_path)))
                print(f"Moved correctly classified video: {video_path} -> {correct_class_path}")
            else:
                print(f"Incorrectly classified video: {video_path}")


if __name__ == "__main__":
    process_dataset()