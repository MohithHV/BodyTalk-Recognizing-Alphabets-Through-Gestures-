import os
import json
import webbrowser
import base64
from threading import Timer

import cv2
import joblib
import mediapipe as mp
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request, send_from_directory
from flask_socketio import SocketIO, emit
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from utils.logger_config import logger

# --- App Initialization ---
logger.info("Initializing Flask App...")
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_very_secret_key'
socketio = SocketIO(app, cors_allowed_origins="*")

# --- Load Config ---
with open("config.json") as f:
    config = json.load(f)

MODELS_DIR = os.path.dirname(config["model_path"])
DATA_DIR = os.path.dirname(config["data_path"])

# --- Global State ---
testing_clients = {}
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
logger.info(f"Model directory set to: {MODELS_DIR}")
logger.info(f"Data directory set to: {DATA_DIR}")

# --- Helper Functions ---
def get_model_path(model_name):
    logger.debug(f"Getting model path for: {model_name}")
    return os.path.join(MODELS_DIR, f"{model_name}.pkl")

def get_data_path(model_name):
    logger.debug(f"Getting data path for: {model_name}")
    return os.path.join(DATA_DIR, f"{model_name}_data.csv")

def get_models():
    logger.debug("Fetching list of models.")
    models = sorted([f.replace(".pkl", "") for f in os.listdir(MODELS_DIR) if f.endswith(".pkl")])
    logger.debug(f"Found models: {models}")
    return models

def get_model_gestures(model_name):
    model_path = get_model_path(model_name)
    logger.debug(f"Fetching gestures for model: {model_name} from {model_path}")
    if not os.path.exists(model_path):
        logger.warning(f"Model file not found: {model_path}")
        return []
    try:
        model = joblib.load(model_path)
        gestures = model.classes_.tolist()
        logger.debug(f"Successfully loaded gestures: {gestures}")
        return gestures
    except Exception as e:
        logger.error(f"Error loading model {model_name}: {e}", exc_info=True)
        return []



# --- Flask Routes ---
@app.route('/')
def index():
    logger.info(f"Serving index.html for request from {request.remote_addr}")
    return render_template('index.html')

@app.route('/api/models', methods=['GET'])
def list_models():
    logger.info("API call to list models.")
    models = get_models()
    logger.debug(f"Returning models: {models}")
    return jsonify(models)

@app.route('/api/model_count', methods=['GET'])
def get_model_count():
    logger.info("API call to get model count.")
    count = len(get_models())
    logger.debug(f"Returning model count: {count}")
    return jsonify(count)

@app.route('/api/models/<string:model_name>', methods=['GET'])
def get_model_details(model_name):
    logger.info(f"API call for details of model: {model_name}")
    gestures = get_model_gestures(model_name)
    logger.debug(f"Returning gestures for {model_name}: {gestures}")
    return jsonify({"gestures": gestures})

@app.route('/api/models/<string:model_name>/download', methods=['GET'])
def download_model(model_name):
    logger.info(f"API call to download model: {model_name}")
    model_path = get_model_path(model_name)
    if not os.path.exists(model_path):
        logger.error(f"Model file not found for download: {model_path}")
        return jsonify({"error": "Model not found"}), 404
    
    try:
        # For demonstration, we're just sending the .pkl file.
        # In a real-world scenario, you would convert this to a .h5 file if needed.
        return send_from_directory(MODELS_DIR, f"{model_name}.pkl", as_attachment=True)
    except Exception as e:
        logger.error(f"Error preparing model for download: {e}", exc_info=True)
        return jsonify({"error": "Could not process model for download"}), 500


# --- WebSocket Events ---
@socketio.on('connect')
def on_connect():
    logger.info(f"Client connected: {request.sid}")

@socketio.on('disconnect')
def on_disconnect():
    if request.sid in testing_clients:
        testing_clients.pop(request.sid)
        logger.info(f"Stopped running test for disconnected client: {request.sid}")
    logger.info(f"Client disconnected: {request.sid}")

client_frames = {}

@socketio.on('stream_frame')
def handle_stream_frame(data):
    sid = request.sid
    try:
        img_bytes = base64.b64decode(data['image'].split(',')[1])
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        client_frames[sid] = frame
    except Exception as e:
        logger.error(f"[{sid}] Error decoding streamed frame: {e}", exc_info=True)

@socketio.on('start_capture')
def handle_capture(data):
    sid = request.sid
    logger.info(f"[{sid}] Received start_capture request with data: {data}")
    gestures = data['gestures']
    model_name = data['model_name']
    is_retraining = data.get('is_retraining', False)
    capture_time = data.get('capture_time', 10)
    capture_time = max(10, min(int(capture_time), 60))

    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    logger.debug(f"[{sid}] MediaPipe Pose initialized.")
    
    captured_data = []
    NUM_LANDMARKS = 33

    try:
        for gesture in gestures:
            logger.debug(f"[{sid}] Starting capture for gesture: {gesture}")
            emit('update_status', {'message': f"Get ready for: {gesture}"})
            socketio.sleep(5)
            logger.debug(f"[{sid}] 5-second preparation time for {gesture} ended.")

            for i in range(3, 0, -1):
                emit('update_status', {'message': f"Starting in {i}..."})
                socketio.sleep(1)
                logger.debug(f"[{sid}] Countdown: {i} for {gesture}")

            emit('update_status', {'message': f"Capturing: {gesture}"})
            start_time = cv2.getTickCount()
            frame_count = 0
            while (cv2.getTickCount() - start_time) / cv2.getTickFrequency() < capture_time:
                frame = client_frames.get(sid)
                if frame is None:
                    logger.warning(f"[{sid}] No frame received from client for {gesture}. Skipping.")
                    socketio.sleep(0.03)
                    continue

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb_frame)

                if results.pose_landmarks:
                    pose_landmarks = results.pose_landmarks
                    # We don't need to draw landmarks on the server side anymore
                    
                    row = [gesture]
                    for lm in pose_landmarks.landmark:
                        row.extend([lm.x, lm.y, lm.z])
                    if len(row) == 1 + 3 * NUM_LANDMARKS:
                        captured_data.append(row)
                        logger.debug(f"[{sid}] Captured landmark data for {gesture}. Total: {len(captured_data)}")
                    else:
                        logger.warning(f"[{sid}] Mismatched landmark count for capture. Expected {1 + 3 * NUM_LANDMARKS}, got {len(row)}.")
                else:
                    logger.debug(f"[{sid}] No pose detected for {gesture} in current frame.")

                socketio.sleep(0.03)
                frame_count += 1
            logger.debug(f"[{sid}] Finished capturing {frame_count} frames for {gesture}. Captured {len(captured_data)} data points for this gesture.")
    except Exception as e:
        logger.error(f"[{sid}] Error during capture process: {e}", exc_info=True)
        emit('update_status', {'message': f'Error during capture: {e}'})
    finally:
        pose.close()
        client_frames.pop(sid, None)
        logger.info(f"[{sid}] MediaPipe pose closed.")

    if not captured_data:
        logger.warning(f"[{sid}] No data was captured for model {model_name}. Skipping training.")
        emit('capture_complete', {'message': f"No data captured for '{model_name}'. Training skipped."})
        return

    columns = ["label"] + [f"{axis}{i}" for i in range(NUM_LANDMARKS) for axis in ["x", "y", "z"]]
    new_df = pd.DataFrame(captured_data, columns=columns)
    data_path = get_data_path(model_name)

    if is_retraining and os.path.exists(data_path):
        logger.info(f"[{sid}] Retraining. Appending to existing data at {data_path}")
        old_df = pd.read_csv(data_path)
        df = pd.concat([old_df, new_df], ignore_index=True)
    else:
        logger.info(f"[{sid}] Creating new dataset at {data_path}")
        df = new_df

    df.to_csv(data_path, index=False)
    logger.info(f"[{sid}] Data saved successfully to {data_path}. Total rows: {len(df)}")
    
    # Emit an event to confirm capture completion and prompt for training
    emit('confirm_training', {'model_name': model_name, 'is_retraining': is_retraining, 'captured_data_count': len(captured_data)})
    logger.info(f"[{sid}] Capture complete. Captured {len(captured_data)} data points for {model_name}. Awaiting training confirmation.")

@socketio.on('start_training')
def handle_training(data):
    sid = request.sid
    model_name = data['model_name']
    is_retraining = data['is_retraining']
    captured_data_count = data['captured_data_count'] # This is just for logging, actual data is handled by files
    logger.info(f"[{sid}] Received start_training request for model: {model_name}, Retraining: {is_retraining}, Captured data points: {captured_data_count}")

    try:
        data_path = get_data_path(model_name)
        if not os.path.exists(data_path):
            logger.error(f"[{sid}] Data file not found for training: {data_path}")
            emit('training_complete', {'message': f"Error: Data for '{model_name}' not found. Please recapture."})
            return

        df = pd.read_csv(data_path)
        logger.info(f"[{sid}] Loaded data for training from {data_path}. Shape: {df.shape}")

        X = df.drop("label", axis=1)
        y = df["label"]
        logger.debug(f"[{sid}] Dataframe shape: {df.shape}, X shape: {X.shape}, y shape: {y.shape}")

        if len(y.unique()) < 2:
            logger.error(f"[{sid}] Not enough classes ({len(y.unique())}) for stratification. Need at least 2.")
            emit('training_complete', {'message': f"Error: Not enough unique gestures ({len(y.unique())}) for training. Need at least 2."})
            return

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        logger.debug(f"[{sid}] Data split into train/test. X_train: {X_train.shape}, X_test: {X_test.shape}")

        emit('update_status', {'message': "Training model..."})
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)
        logger.info(f"[{sid}] Model training complete.")

        joblib.dump(clf, get_model_path(model_name))
        logger.info(f"[{sid}] Model saved to {get_model_path(model_name)}")
        emit('training_complete', {'message': f"Model '{model_name}' trained successfully!"})
    except Exception as e:
        logger.error(f"[{sid}] Error during training process: {e}", exc_info=True)
        emit('training_complete', {'message': f'Error during training: {e}'})

@socketio.on('start_training')
def handle_training(data):
    sid = request.sid
    model_name = data['model_name']
    is_retraining = data['is_retraining']
    captured_data_count = data['captured_data_count'] # This is just for logging, actual data is handled by files
    logger.info(f"[{sid}] Received start_training request for model: {model_name}, Retraining: {is_retraining}, Captured data points: {captured_data_count}")

    try:
        data_path = get_data_path(model_name)
        if not os.path.exists(data_path):
            logger.error(f"[{sid}] Data file not found for training: {data_path}")
            emit('training_complete', {'message': f"Error: Data for '{model_name}' not found. Please recapture."})
            return

        df = pd.read_csv(data_path)
        logger.info(f"[{sid}] Loaded data for training from {data_path}. Shape: {df.shape}")

        X = df.drop("label", axis=1)
        y = df["label"]
        logger.debug(f"[{sid}] Dataframe shape: {df.shape}, X shape: {X.shape}, y shape: {y.shape}")

        if len(y.unique()) < 2:
            logger.error(f"[{sid}] Not enough classes ({len(y.unique())}) for stratification. Need at least 2.")
            emit('training_complete', {'message': f"Error: Not enough unique gestures ({len(y.unique())}) for training. Need at least 2."})
            return

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        logger.debug(f"[{sid}] Data split into train/test. X_train: {X_train.shape}, X_test: {X_test.shape}")

        emit('update_status', {'message': "Training model..."})
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)
        logger.info(f"[{sid}] Model training complete.")

        joblib.dump(clf, get_model_path(model_name))
        logger.info(f"[{sid}] Model saved to {get_model_path(model_name)}")
        emit('training_complete', {'message': f"Model '{model_name}' trained successfully!"})
    except Exception as e:
        logger.error(f"[{sid}] Error during training process: {e}", exc_info=True)
        emit('training_complete', {'message': f'Error during training: {e}'})

@socketio.on('retake_data')
def handle_retake_data(data):
    sid = request.sid
    model_name = data['model_name']
    logger.info(f"[{sid}] Received retake_data request for model: {model_name}. User chose to retake data.")
    emit('retake_data_confirmed', {'message': f"Retaking data for {model_name}."})


@socketio.on('start_test')
def handle_test(data):
    sid = request.sid
    model_name = data['model_name']
    logger.info(f"[{sid}] Received start_test request for model: {model_name}")
    model_path = get_model_path(model_name)

    if not os.path.exists(model_path):
        logger.error(f"[{sid}] Model not found for testing: {model_path}")
        emit('update_status', {'message': f"Error: Model {model_name} not found."})
        return

    try:
        model = joblib.load(model_path)
        logger.debug(f"[{sid}] Model {model_name} loaded for testing.")
    except Exception as e:
        logger.error(f"[{sid}] Error loading model {model_name} for test: {e}", exc_info=True)
        emit('update_status', {'message': f'Error loading model for test: {e}'})
        return

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    logger.debug(f"[{sid}] MediaPipe Pose initialized for testing.")
    
    testing_clients[sid] = True
    emit('update_status', {'message': "Live test started. Show a gesture!"})
    logger.info(f"[{sid}] Starting test loop.")

    try:
        while testing_clients.get(sid):
            frame = client_frames.get(sid)
            if frame is None:
                logger.warning(f"[{sid}] No frame received from client during test. Skipping.")
                socketio.sleep(0.03)
                continue

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)

            prediction_text = "No Pose Detected"
            confidence_text = ""

            if results.pose_landmarks:
                pose_landmarks = results.pose_landmarks
                # We don't need to draw landmarks on the server side anymore
                
                row = []
                for lm in pose_landmarks.landmark:
                    row.extend([lm.x, lm.y, lm.z])
                
                if len(row) == 3 * 33: # 3 (x,y,z) * 33 landmarks
                    X_live = pd.DataFrame([row])
                    try:
                        prediction = model.predict(X_live)[0]
                        confidence = np.max(model.predict_proba(X_live))
                        prediction_text = f"{prediction}"
                        confidence_text = f"({confidence:.2f})"
                        logger.debug(f"[{sid}] Prediction: {prediction_text} {confidence_text}")
                    except Exception as e:
                        logger.error(f"[{sid}] Error during model prediction: {e}", exc_info=True)
                        prediction_text = "Prediction Error"
                else:
                    logger.warning(f"[{sid}] Mismatched landmark count for prediction. Expected {3*33}, got {len(row)}.")
                    prediction_text = "Invalid Data"

            emit('prediction', {'gesture': prediction_text, 'confidence': confidence_text})
            socketio.sleep(0.03)
    except Exception as e:
        logger.error(f"[{sid}] Unhandled error in test loop: {e}", exc_info=True)
        emit('update_status', {'message': f'Error during test: {e}'})
    finally:
        pose.close()
        client_frames.pop(sid, None)
        logger.info(f"[{sid}] Test loop stopped and MediaPipe pose closed.")
    emit('update_status', {'message': "Test stopped."})

@socketio.on('retrain_model')
def handle_retrain_model(data):
    sid = request.sid
    model_name = data['model_name']
    retrain_option = data['retrain_option'] # 'same_gestures' or 'new_gestures'
    capture_time = data.get('capture_time', 10) # Get capture time, default to 10
    logger.info(f"[{sid}] Received retrain_model request for model: {model_name} with option: {retrain_option}")

    try:
        if retrain_option == 'same_gestures':
            gestures = get_model_gestures(model_name)
            if not gestures:
                logger.error(f"[{sid}] No gestures found for model {model_name}. Cannot retrain with same gestures.")
                emit('retrain_error', {'message': f"Error: No gestures found for model {model_name}."})
                return
            emit('start_capture_for_retrain', {'model_name': model_name, 'gestures': gestures, 'is_retraining': True, 'capture_time': capture_time})
            logger.info(f"[{sid}] Initiating recapture for retraining model {model_name} with existing gestures: {gestures}")
        elif retrain_option == 'new_gestures':
            # Frontend will prompt for new gestures and then call start_capture
            emit('prompt_for_new_gestures', {'model_name': model_name})
            logger.info(f"[{sid}] Prompting user for new gestures for retraining model {model_name}.")
        else:
            logger.error(f"[{sid}] Invalid retrain option: {retrain_option}")
            emit('retrain_error', {'message': "Invalid retraining option."})

    except Exception as e:
        logger.error(f"[{sid}] Error handling retrain_model request: {e}", exc_info=True)
        emit('retrain_error', {'message': f'Error during retraining process: {e}'})

@socketio.on('delete_model')
def handle_delete_model(data):
    sid = request.sid
    model_name = data['model_name']
    logger.info(f"[{sid}] Received delete_model request for model: {model_name}")
    try:
        model_path = get_model_path(model_name)
        data_path = get_data_path(model_name)

        if os.path.exists(model_path):
            os.remove(model_path)
            logger.info(f"[{sid}] Deleted model file: {model_path}")
        else:
            logger.warning(f"[{sid}] Model file not found for deletion: {model_path}")

        if os.path.exists(data_path):
            os.remove(data_path)
            logger.info(f"[{sid}] Deleted data file: {data_path}")
        else:
            logger.warning(f"[{sid}] Data file not found for deletion: {data_path}")

        emit('model_deleted', {'message': f"Model '{model_name}' and its data deleted successfully."})
        logger.info(f"[{sid}] Model {model_name} and associated data successfully deleted.")
    except Exception as e:
        logger.error(f"[{sid}] Error deleting model {model_name}: {e}", exc_info=True)
        emit('delete_error', {'message': f'Error deleting model: {e}'})

@socketio.on('stop_test')
def stop_test():
    sid = request.sid
    if sid in testing_clients:
        testing_clients.pop(sid)
    logger.info(f"Test stopped via explicit request for client: {sid}")

def open_browser():
    logger.info("Opening web browser.")
    webbrowser.open_new('http://127.0.0.1:5000/')

if __name__ == '__main__':
    logger.info("Starting application server.")
    Timer(1, open_browser).start()
    socketio.run(app, port=5000, allow_unsafe_werkzeug=True)
    logger.info("Application server has stopped.")