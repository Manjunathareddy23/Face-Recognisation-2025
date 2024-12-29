import streamlit as st
import cv2
import face_recognition
import numpy as np
import pandas as pd
import os
from datetime import datetime

# Initialize attendance data in Streamlit session state
if 'attendance' not in st.session_state:
    st.session_state.attendance = []

st.title("Face Recognition Attendance System")

# Function to load known faces and their encodings
def load_known_faces(directory="known_faces"):
    known_encodings = []
    known_names = []

    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(directory, filename)
            image = face_recognition.load_image_file(img_path)
            encoding = face_recognition.face_encodings(image)[0]

            known_encodings.append(encoding)
            known_names.append(os.path.splitext(filename)[0])

    return known_encodings, known_names

# Load known face encodings and their names
known_face_encodings, known_face_names = load_known_faces()

# Function to recognize faces in a given frame
def recognize_faces(frame, known_encodings, known_names):
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    names = []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        name = "Unknown"

        if True in matches:
            first_match_index = matches.index(True)
            name = known_names[first_match_index]

        names.append(name)

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

        # Convert BGR to RGB for face_recognition
        rgb_frame = frame[:, :, ::-1]

        # Detect and recognize faces
        face_locations, names = recognize_faces(rgb_frame, known_face_encodings, known_face_names)

        # Draw rectangles and labels on the frame
        for (top, right, bottom, left), name in zip(face_locations, names):
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)  # Rectangle around the face
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)  # Name label
            mark_attendance(name)

        # Display the frame in Streamlit
        stframe.image(frame, channels="BGR")

    video_capture.release()

# Display attendance log
st.subheader("Attendance Log")
if st.session_state.attendance:
    df = pd.DataFrame(st.session_state.attendance)
    st.dataframe(df)
else:
    st.write("No attendance recorded yet.")
