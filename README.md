Deepfake Detection System

An AI-powered deepfake detection application using deep learning and Streamlit. This system detects manipulated or fake videos using trained neural networks to classify videos as **Real** or **Fake**.

🎯 Features

✅ **User Authentication** - Secure login and registration system  
✅ **Video Upload** - Support for MP4, AVI, and MOV formats  
✅ **Frame Extraction** - Intelligent frame sampling from videos  
✅ **Deep Learning Analysis** - TensorFlow/Keras-based detection model  
✅ **Real-time Predictions** - Fast classification with confidence scores  
✅ **Database Logging** - Stores user predictions for history tracking  
✅ **Intuitive UI** - User-friendly Streamlit interface  

🔍 How It Works

1. Upload a Video - Users select a video file from their device
2. Frame Extraction - System extracts 10 key frames from the video
3. Preprocessing - Frames are resized to 128x128 and normalized
4. Model Analysis - Deep learning model analyzes frame patterns
5. Classification - Outputs prediction (Real/Fake) with confidence percentage
6. Result Storage - Predictions are saved to database for future reference

 📋 Installation

 Prerequisites
- Python 3.8 or higher
- pip or conda package manager
- 2GB disk space (for model and dependencies)

 Setup Instructions

1. Clone the repository

git clone https://github.com/Bhoomika-T-Raj/Fake-Video-Detection.git
cd Fake-Video-Detection

2. Create a virtual environment

python -m venv venv

Activate virtual environment
1.On Windows:
venv\Scripts\activate

2.On macOS/Linux:
source venv/bin/activate

3.Install dependencies

pip install -r requirements.txt

Required packages:

streamlit
tensorflow
keras
opencv-python
numpy

pip install streamlit tensorflow keras opencv-python numpy sqlite3

🚀 Usage
Run the Streamlit Application

cd "Fake Video Detection"
streamlit run app.py

The app will open in your browser at "http://localhost:8501"

Features:
Login/Register - Create an account or login with existing credentials
Upload Video - Select a video file to analyze
Get Prediction - Receive real/fake classification with confidence score
View History - Check past predictions from your account

📁 Project Structure
Fake-Video-Detection/
├── Fake Video Detection/
│   ├── app.py                          # Main Streamlit application
│   ├── predict.py                      # Prediction script
│   ├── train.py                        # Model training script
│   ├── train2.py                       # Alternative training script
│   ├── fake_video_detection_with_plot.py
│   ├── deepfake_detection_model.keras  # Trained model
│   ├── users.db                        # SQLite database
│   ├── coverpage.png                   # UI cover image
│   ├── Coverside.png                   # Sidebar image
│   ├── DFVside.png                     # Sidebar image
│   ├── Dataset/
│   │   ├── fake/                       # Fake videos for training
│   │   └── real/                       # Real videos for training
│   ├── testing_videos/
│   │   ├── fake/                       # Test fake videos
│   │   └── real/                       # Test real videos
│   ├── *.ipynb                         # Jupyter notebooks
├── myapp/                              # (Optional) React frontend
├── README.md
├── .gitignore
└── requirements.txt

🧠 Model Details

Architecture: Convolutional Neural Network (CNN)
Framework: TensorFlow/Keras
Input Size: 128x128 RGB frames
Output: Binary classification (Real/Fake)
Training Data: Custom dataset of real and deepfake videos

Model Performance:
Accuracy on test set: 92%
Training frames per video: 10
Processing time: ~2-5 seconds per video

📊 Dataset
Real Videos: Collection of authentic video samples
Fake Videos: Deepfake videos generated using various deepfaking techniques
Format: MP4, AVI, MOV
Location: Dataset/real/ and Dataset/fake/

🔐 Security
User passwords are stored in SQLite database (implement hashing in production)
Session-based authentication
Video files are processed in temporary directories
Predictions are associated with user accounts
⚠️ For production use:

Implement bcrypt or similar for password hashing
Use a production database (PostgreSQL, MySQL)
Add HTTPS support
Implement rate limiting

📝 Notebooks
The project includes Jupyter notebooks for exploration:

Training fake videos.ipynb - Model training workflow
Processing fake videos.ipynb - Data preprocessing pipeline
Fake Video detection with plot.ipynb - Detection with visualization
🤝 Contributing
Contributions are welcome! To contribute:

Fork the repository
Create a feature branch (git checkout -b feature/AmazingFeature)
Commit changes (git commit -m 'Add AmazingFeature')
Push to branch (git push origin feature/AmazingFeature)
Open a Pull Request
📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

⚖️ Disclaimer
This deepfake detection system is providedfor educational and research purposes. While the model aims for high accuracy, it may not catch all deepfakes. Users should not rely solely on this tool for critical applications. Always verify video authenticity through multiple methods.

👤 Author
Bhoomika T Raj
GitHub: @Bhoomika-T-Raj

📞 Support
For issues, questions, or suggestions:

Open an Issue on GitHub
Contact: bhoomikatraj564@gmail.com

🎓 Technologies Used

Python 3.8+
TensorFlow/Keras - Deep learning
OpenCV - Video processing
Streamlit - Web UI
SQLite - Database
NumPy - Numerical computing

🔮 Future Improvements

 Add support for more video formats
 Implement real-time video stream analysis
 Add model explainability (LIME/SHAP)
 Deploy on cloud (Heroku, AWS, Google Cloud)
  Implement advanced ensemble models
 Add multi-language support
 Improve model accuracy with more training data

 ⭐ If you find this project helpful, please consider giving it a star!

