import csv
import time
from datetime import datetime
import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, render_template, Response, redirect, url_for, request, send_from_directory, flash
import os
import plotly.express as px
import pandas as pd
from werkzeug.utils import secure_filename
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import json

# Flask app setup
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.secret_key = 'supersecretkey'

# Load the ViT-B16 model
def load_vit_model(weights_path, num_classes, device):
    model = models.vit_b_16(pretrained=False)  # Initialize ViT-B16
    num_ftrs = model.heads.head.in_features
    model.heads.head = nn.Linear(num_ftrs, num_classes)  # Adjust the final layer
    checkpoint = torch.load(weights_path, map_location=device)
    model.load_state_dict(checkpoint['MODEL_STATE'] if 'MODEL_STATE' in checkpoint else checkpoint)
    return model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights_path = './best_model.pth'
num_classes = 5  # Number of gesture classes
model = load_vit_model(weights_path, num_classes, device).to(device)
model.eval()

# MediaPipe Hands setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Default labels dictionary
labels_dict = {0: 'fist', 1: 'ok', 2: 'peace', 3: 'stop', 4: 'two up'}
custom_labels_dict = labels_dict.copy()

# Image preprocessing transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Initialize variables for tracking gestures
previous_gesture = None
gesture_start_time = None
gesture_data_list = []
capture_flag = True  # This flag is used to indicate when to capture
start_recording_time = None  # To record the start time of the session

# Default labels dictionary
labels_dict = {0: 'fist', 1: 'ok', 2: 'peace', 3: 'stop', 4: 'two up'}
custom_labels_dict = labels_dict.copy()  # To store custom labels set by user

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/set_labels', methods=['GET', 'POST'])
def set_labels():
    global custom_labels_dict
    if request.method == 'POST':
        custom_labels_dict[0] = request.form['label1']
        custom_labels_dict[1] = request.form['label2']
        custom_labels_dict[2] = request.form['label3']
        custom_labels_dict[3] = request.form['label4']
        custom_labels_dict[4] = request.form['label5']
        # Remove empty labels
        custom_labels_dict = {k: v for k, v in custom_labels_dict.items() if v}
        return redirect(url_for('recognize'))
    return render_template('set_labels.html')

@app.route('/recognize')
def recognize():
    return render_template('recognize.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_frames():
    global previous_gesture, gesture_start_time, gesture_data_list, capture_flag, start_recording_time

    # Initialize start recording time
    start_recording_time = datetime.now()
    cap = cv2.VideoCapture(0)

    while capture_flag:
        ret, frame = cap.read()
        if not ret:
            break

        H, W, _ = frame.shape  # Frame dimensions
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB for Mediapipe

        # Process the frame for hand landmarks
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            # Draw skeletons and compute bounding box
            x_ = []
            y_ = []
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw the hand skeleton
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
                # Collect coordinates for bounding box calculation
                for landmark in hand_landmarks.landmark:
                    x_.append(landmark.x)
                    y_.append(landmark.y)

            # Calculate bounding box
            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10
            x2 = int(max(x_) * W) + 10
            y2 = int(max(y_) * H) + 10

            # Crop and preprocess the frame for ViT model
            cropped_frame = frame_rgb[y1:y2, x1:x2]  # Crop the bounding box region
            if cropped_frame.size > 0:  # Ensure the cropped region is valid
                cropped_image = Image.fromarray(cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB))  # Convert to PIL
                input_tensor = transform(cropped_image).unsqueeze(0).to(device)  # Preprocess

                # Perform gesture recognition using ViT model
                with torch.no_grad():
                    outputs = model(input_tensor)
                    _, predicted = torch.max(outputs, 1)
                    detected_gesture = custom_labels_dict.get(predicted.item(), None)

                # Draw the detected gesture and bounding box
                if detected_gesture:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)  # Draw bounding box
                    cv2.putText(frame, detected_gesture, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

                    # Track gesture changes
                    current_time = datetime.now()
                    relative_time = current_time - start_recording_time

                    if detected_gesture != previous_gesture:
                        if previous_gesture is not None:
                            # Calculate gesture duration
                            gesture_end_time = relative_time.total_seconds()
                            gesture_duration = gesture_end_time - gesture_start_time
                            # Store gesture data
                            gesture_data_list.append([previous_gesture, gesture_start_time, gesture_end_time, round(gesture_duration, 2)])

                        previous_gesture = detected_gesture
                        gesture_start_time = relative_time.total_seconds()

        # Encode the frame for streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()



@app.route('/save_data')
def save_gesture_data():
    global capture_flag
    capture_flag = False

    # Ensure gesture data is actually populated
    print("Saving gesture data:", gesture_data_list)

    # Ensure the static directory exists
    os.makedirs('static', exist_ok=True)

    # Save data to JSON file in Label Studio-compatible format
    json_file_path = os.path.join('static', 'gesture_data_labelstudio.json')
    save_label_studio_json(gesture_data_list, json_file_path)

    # Save data to CSV file for visualization
    csv_file_path = os.path.join('static', 'gesture_data.csv')
    save_gesture_csv(gesture_data_list, csv_file_path)

    return redirect(url_for('data'))

import random  # Make sure to import the random module
import uuid  # Make sure to import uuid for unique IDs
from datetime import datetime  # Import datetime for timestamp

import string  # Ensure this is at the top of your script
import random  # Ensure random is also imported

def generate_alphanumeric_id(length=5):
    """Generates a random alphanumeric ID."""
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))


def save_label_studio_json(gesture_data, file_path):
    current_time = datetime.utcnow().isoformat() + "Z"
    
    # Create a single task with all annotations
    annotations = {
        "id": 1,  # Task ID
        "annotations": [
            {
                "id": 1,  # Annotation ID
                "completed_by": 1,
                "result": [],
                "was_cancelled": False,
                "ground_truth": False,
                "created_at": current_time,
                "updated_at": current_time,
                "draft_created_at": current_time,
                "lead_time": sum(duration for _, _, _, duration in gesture_data),
                "prediction": {},
                "result_count": 0,
                "unique_id": str(uuid.uuid4()),
                "import_id": None,
                "last_action": None,
                "task": 1,
                "project": 25,
                "updated_by": 1,
                "parent_prediction": None,
                "parent_annotation": None,
                "last_created_by": None
            }
        ],
        "file_upload": "1212df4d-HandyLabels.MP4",
        "drafts": [],
        "predictions": [],
        "data": {
            "video_url": "/data/upload/30/030cca83-Video_1.mp4"
        },
        "meta": {},
        "created_at": current_time,
        "updated_at": current_time,
        "inner_id": 1,
        "total_annotations": 1,
        "cancelled_annotations": 0,
        "total_predictions": 0,
        "comment_count": 0,
        "unresolved_comment_count": 0,
        "last_comment_updated_at": None,
        "project": 25,
        "updated_by": 1,
        "comment_authors": []
    }

    # Add each gesture as an individual result within the annotation
    for gesture, start_time, end_time, duration in gesture_data:
        annotation_result = {
            "original_length": end_time - start_time,
            "value": {
                "start": start_time,
                "end": end_time,
                "channel": 0,
                "labels": [gesture]
            },
            "id": generate_alphanumeric_id(),  # Generate a unique 5-character alphanumeric ID for each result
            "from_name": "tricks",
            "to_name": "audio",
            "type": "labels",
            "origin": "manual"
        }
        annotations["annotations"][0]["result"].append(annotation_result)

    # Save the consolidated JSON to the file
    with open(file_path, 'w') as json_file:
        json.dump([annotations], json_file, indent=2)

    print(f"Label Studio JSON saved to: {file_path}")


def save_gesture_csv(gesture_data, file_path):
    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Gesture', 'Start Time', 'End Time', 'Duration'])
        for gesture, start_time, end_time, duration in gesture_data:
            writer.writerow([gesture, start_time, end_time, duration])

@app.route('/data')
def data():
    gesture_data = load_csv_data()

    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(gesture_data, columns=['Gesture', 'Start Time', 'End Time', 'Duration'])

    # Count occurrences of each gesture
    gesture_counts = df['Gesture'].value_counts().reset_index()
    gesture_counts.columns = ['Gesture', 'Count']

    # Create the pie chart using Plotly
    fig = px.pie(gesture_counts, values='Count', names='Gesture', title='Gesture Distribution')

    # Convert the plotly chart to HTML
    html_chart = fig.to_html(full_html=False)

    return render_template('data.html', gesture_data=gesture_data, html_chart=html_chart)

def load_csv_data():
    gesture_data = []
    with open('static/gesture_data.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip the header row
        for row in reader:
            gesture_data.append(row)
    return gesture_data

@app.route('/download_json')
def download_json():
    file_path = os.path.join('static', 'gesture_data_labelstudio.json')

    if not os.path.isfile(file_path):
        return "JSON file not found!", 404

    return send_from_directory('static', 'gesture_data_labelstudio.json', as_attachment=True)

@app.route('/download_csv')
def download_csv():
    file_path = os.path.join('static', 'gesture_data.csv')

    if not os.path.isfile(file_path):
        return "CSV file not found!", 404

    return send_from_directory('static', 'gesture_data.csv', as_attachment=True)

# Import Data Functionality to Visualize Imported CSV
@app.route('/import_data', methods=['GET', 'POST'])
def import_data():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            return redirect(url_for('visualize_data', file_path=file_path))
    return render_template('import_data.html')

@app.route('/visualize_data')
def visualize_data():
    file_path = request.args.get('file_path')

    if not os.path.exists(file_path):
        return "The file could not be found.", 404

    return visualize_csv(file_path)

def visualize_csv(file_path):
    try:
        # Load gesture data from CSV and process it for visualization
        data = pd.read_csv(file_path)

        # Check if necessary columns are present
        required_columns = ['Gesture', 'Start Time', 'End Time', 'Duration']
        if not set(required_columns).issubset(data.columns):
            return f"The uploaded CSV must contain the following columns: {required_columns}", 400

        # Extract relevant columns
        gesture_df = data[required_columns]

        # Generate a pie chart for gesture distribution
        gesture_counts = gesture_df['Gesture'].value_counts().reset_index()
        gesture_counts.columns = ['Gesture', 'Count']

        # Create the pie chart using Plotly
        fig = px.pie(gesture_counts, values='Count', names='Gesture', title='Gesture Distribution')

        # Convert the plotly chart to HTML
        html_chart = fig.to_html(full_html=False)

        # Render the data.html template with the gesture data and chart
        return render_template('data.html', gesture_data=gesture_df.to_dict('records'), html_chart=html_chart)

    except Exception as e:
        return f"An error occurred while processing the file: {str(e)}", 500

if __name__ == '__main__':
    app.run(debug=True)
