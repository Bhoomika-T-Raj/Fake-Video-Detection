import os
import cv2
import numpy as np
import tensorflow as tf
import tempfile
import sqlite3
import streamlit as st
from tensorflow.keras.models import load_model

# Configuration
MODEL_PATH = "deepfake_detection_model.keras"
FRAME_COUNT = 10
FRAME_SIZE = (128, 128)

# Database Setup
def init_db():
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE,
                    password TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT,
                    video TEXT,
                    result TEXT,
                    confidence REAL)''')
    conn.commit()
    conn.close()

init_db()

st.image("coverpage.png")
st.sidebar.image("Coverside.png")
st.sidebar.markdown("""
# **Deepfake Detection System**
### **AI-Powered Solution for Detecting Deepfake Videos**
Welcome to our **Deepfake Detection System**, an advanced AI-driven platform designed to detect manipulated or fake 
videos using deep learning models. With the rise of deepfake technology, it has become crucial to have robust tools 
that can differentiate between real and fake content to prevent misinformation and malicious activities.

## **🔍 How It Works**
1. **Upload a Video** - Users can upload any video in MP4, AVI, or MOV format.
2. **Frame Extraction** - The system extracts key frames from the video.
3. **Deep Learning Model Analysis** - A trained AI model analyzes the frames to detect inconsistencies.
4. **Real vs. Fake Prediction** - The system classifies the video as **Real** or **Fake** along with a confidence score.

## **🛠 Key Features**
✔️ **User Authentication** - Secure login and registration system.  
✔️ **Deepfake Detection** - AI-based model trained on high-quality datasets.  
✔️ **Fast & Accurate** - Provides real-time analysis and predictions.  
✔️ **Confidence Score** - Ensures transparency in the classification process.  
✔️ **Database Logging** - Saves user predictions for future reference.  

## **🔒 Why Deepfake Detection Matters?**
- **Prevents Misinformation** - Identifies fake videos before they spread.  
- **Enhances Security** - Protects individuals and organizations from deception.  
- **Safeguards Digital Integrity** - Ensures the authenticity of video content.  

## **🚀 Get Started**
1. **Register/Login** to access the deepfake detection tool.  
2. **Upload a video** for analysis.  
3. **Get an instant prediction** with confidence scores.  

---
🎯 **Developed using TensorFlow, OpenCV, and Streamlit**  
📌 **Secure, Reliable, and Efficient Deepfake Detection**
""")

# Load the trained model
@st.cache_resource
def load_trained_model():
    model = load_model(MODEL_PATH)
    return model

model = load_trained_model()

# Function to extract frames from video
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
        return "Error: Video does not have enough frames for prediction."
    frames = preprocess_frames(frames)
    predictions = model.predict(frames)
    predicted_class = np.argmax(predictions, axis=1)[0]
    class_labels = {0: "video is Fake", 1: "video is Real"}
    confidence = predictions[0][predicted_class]
    return class_labels[predicted_class], confidence


# User authentication functions
def register_user(username, password):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        conn.close()
        return False


def login_user(username, password):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username = ? AND password = ?", (username, password))
    user = c.fetchone()
    conn.close()
    return user is not None


# Save predictions to database
def save_prediction(username, video_name, result, confidence):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("INSERT INTO predictions (username, video, result, confidence) VALUES (?, ?, ?, ?)",
              (username, video_name, result, confidence))
    conn.commit()
    conn.close()


# Streamlit UI
def main():
    st.title("Deepfake Detection System")
    st.write(
        "Welcome to our AI-powered Deepfake Detection System. "
        "This system helps identify fake videos using deep learning models.")

    menu = ["Home", "Login", "Register"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.subheader("About the System")
        st.write("""
        In recent years, deepfake technology has advanced significantly, 
        making it possible to generate highly realistic fake videos
        that can be used for malicious purposes. 
        Our system provides an AI-powered solution to detect deepfakes accurately and reliably.

        **How It Works:**
        - Upload a video file.
        - Our system extracts frames from the video and processes them using a trained deep learning model.
        - The model then classifies the video as Real or Fake.
        - The prediction and confidence score are displayed to the user.
        """)

    elif choice == "Register":
        st.subheader("Create a New Account")
        new_user = st.text_input("Username")
        new_password = st.text_input("Password", type="password")
        if st.button("Register"):
            if register_user(new_user, new_password):
                st.success("Registration Successful! You can now log in.")
            else:
                st.error("Username already exists. Try a different one.")

    elif choice == "Login":
        st.subheader("Login to Your Account")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if login_user(username, password):
                st.session_state["logged_in"] = True
                st.session_state["username"] = username
                st.success("Login Successful!")
            else:
                st.error("Invalid username or password.")

    if "logged_in" in st.session_state and st.session_state["logged_in"]:
        st.subheader("Deepfake Detection")
        uploaded_file = st.file_uploader("Choose a video file...", type=["mp4", "avi", "mov"])

        if uploaded_file is not None:
            st.video(uploaded_file)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name

            with st.spinner("Processing the video and making predictions..."):
                result, confidence = predict_video_class(tmp_path)

            st.subheader("Prediction Result")

            # Display prediction in large text
            st.markdown(f"""
                <h2 style="text-align: center; color: {'red' if result == 'video is Fake' else 'green'};">
                    Prediction: {result}
                </h2>
                <h3 style="text-align: center; color: black;">
                    Confidence Score: {confidence:.2f}
                </h3>
            """, unsafe_allow_html=True)

            save_prediction(st.session_state["username"], uploaded_file.name, result, confidence)
            os.remove(tmp_path)


if __name__ == "__main__":
    main()
