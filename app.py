import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
from datetime import datetime

# Initialize attendance data in Streamlit session state
if 'attendance' not in st.session_state:
    st.session_state.attendance = []

st.title("Face Recognition Attendance System")

# Initialize mediapipe face detection model
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Function to load known faces (using filenames as identifiers)
def load_known_faces(directory="known_faces"):
    known_faces = []
    known_names = []

    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            # Use filename (without extension) as the name
            known_faces.append(os.path.splitext(filename)[0])
    
    return known_faces

# Load known faces' names
known_face_names = load_known_faces()

# Function to recognize faces using mediapipe
def recognize_faces(frame, known_names):
    face_locations = []
    names = []

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Use mediapipe to detect faces in the frame
    with mp_face_detection.FaceDetection(min_detection_confidence=0.2) as face_detection:
        results = face_detection.process(rgb_frame)
        
        if results.detections:
            for detection in results.detections:
                # Get bounding box coordinates
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                face_locations.append((y, x + w, y + h, x))  # Format: (top, right, bottom, left)

                # Assign name (placeholder logic for demo purposes)
                names.append("Unknown")  # You can replace this with a more sophisticated face recognition method
    
    return face_locations, names

# Function to mark attendance
def mark_attendance(name):
    if name != "Unknown" and name not in [entry['name'] for entry in st.session_state.attendance]:
        now = datetime.now()
        st.session_state.attendance.append({"name": name, "time": now.strftime("%H:%M:%S")})
        st.success(f"{name} marked present at {now.strftime('%H:%M:%S')}")

# Sidebar for options
st.sidebar.header("Options")
camera_enabled = st.sidebar.checkbox("Enable Camera")

if camera_enabled:
    video_capture = cv2.VideoCapture(0)  # Start video capture
    stframe = st.empty()  # Placeholder for video frames

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # Detect and recognize faces
        face_locations, names = recognize_faces(frame, known_face_names)

        # Draw rectangles and labels on the frame
        for (top, right, bottom, left), name in zip(face_locations, names):
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)  # Rectangle around the face
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)  # Name label
            mark_attendance(name)

        # Display the frame in Streamlit
        stframe.image(frame, channels="BGR", use_column_width=True)

    # Release the video capture after loop ends
    video_capture.release()

# Display attendance log
st.subheader("Attendance Log")
if st.session_state.attendance:
    df = pd.DataFrame(st.session_state.attendance)
    st.dataframe(df)
else:
    st.write("No attendance recorded yet.")
