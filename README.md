Overview
This project is a real-time alphabet gesture recognition system that uses body parts (hands, arms, head, torso positions) to classify gestures representing A–Z alphabets. By leveraging computer vision and machine learning, the system captures your body movements from a webcam, detects posture-based gestures, and maps them to corresponding letters of the alphabet.

The application is built using Flask for the web interface, OpenCV for video processing, and a pre-trained machine learning model for classification. It supports real-time classification as well as a data collection mode for adding new body gesture samples.

Features
Real-Time Alphabet Classification: Recognizes and classifies alphabet gestures using your body parts from a live webcam feed.

Full Body Part Tracking: Uses key body joints (hands, arms, shoulders, head, etc.) to detect posture patterns.

Web Interface: Simple, interactive web app to view camera feed and classification results.

Retrainable Model: Add new alphabet gesture samples and retrain the model to improve accuracy.

Data Collection Mode: Collect labeled body gesture data for expanding the dataset.

How It Works
Video Capture: Captures live video from your webcam using OpenCV.

Body Pose Detection: Uses MediaPipe Pose to track key body landmarks.

Feature Extraction: Converts landmark coordinates into features suitable for model input.

Alphabet Classification: A pre-trained model maps the extracted features to corresponding alphabets.

Web Display: A Flask-based interface shows the live feed with classification overlays.


Installation
Clone the repository:
git clone <repository-url>
cd body_alphabet_gesture_classifier

Create and activate a virtual environment:
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

Install dependencies:
pip install -r requirements.txt

Usage
Run the application:
python app.py

Modes:

Classification Mode: Detects and classifies gestures into alphabets in real time.

Data Collection Mode: Record new alphabet gesture samples for model retraining.

Configuration
Adjust parameters in config.json:

model_name: Model file to use (.pkl).

data_file_name: CSV file for collected data.

num_classes: Number of alphabet gestures supported.

Dependencies
Listed in requirements.txt:

Flask

OpenCV-Python

scikit-learn

numpy

mediapipe

Project Structure
├── app.py                  # Main Flask application
├── requirements.txt        # Dependencies
├── config.json             # Settings
├── data/                   # Collected body gesture CSV files
├── models/                 # Trained ML models (.pkl)
├── static/                 # CSS, JS
├── templates/              # HTML templates
└── utils/                  # Helper scripts
