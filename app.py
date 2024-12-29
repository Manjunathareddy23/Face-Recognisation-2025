import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime
import os

# Ensure the 'known_faces' folder exists
known_faces_dir = "known_faces"

# Function to load the known faces dynamically
def load_known_faces(directory="known_faces"):
    known_face_encodings = []
    known_face_names = []
    try:
        # Load the image and encode the face
        image_filename = "manju.jpeg"  # Example image
        image_path = os.path.join(directory, image_filename)
        if os.path.exists(image_path):
            image = face_recognition.load_image_file(image_path)
            encoding = face_recognition.face_encodings(image)[0]
            known_face_encodings.append(encoding)
            known_face_names.append("Manju")
        else:
            st.error(f"Image {image_path} not found.")
    except Exception as e:
        st.error(f"Error loading known faces: {str(e)}")

    return known_face_encodings, known_face_names

# Load known faces
known_face_encodings, known_faces_names = load_known_faces()

# Initialize CSV writing for attendance log
now = datetime.now()
current_date = now.strftime("%Y-%m-%d")
attendance_file = current_date + '.csv'

# Open CSV file for writing attendance (using 'a' mode to append to the file)
def mark_attendance(name):
    with open(attendance_file, 'a', newline='') as f:
        lnwriter = csv.writer(f)
        current_time = now.strftime("%H:%M:%S")
        lnwriter.writerow([name, current_time])

# Face Recognition Processor Class
class FaceRecognitionProcessor(VideoProcessorBase):
    def __init__(self):
        self.present_students = []

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        # Convert the frame to a numpy array
        img = frame.to_ndarray(format="bgr24")
        # Resize the frame for faster processing
        small_frame = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]  # Convert it from BGR to RGB

        # Detect faces in the image
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        face_names = []

        # Compare each detected face with the known faces
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = ""
            face_distance = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distance)
            if matches[best_match_index]:
                name = known_faces_names[best_match_index]

            face_names.append(name)
            if name in known_faces_names:
                if name not in self.present_students:
                    self.present_students.append((name, now.strftime("%H:%M:%S")))
                    mark_attendance(name)  # Log the attendance to CSV
                    st.write(f"{name} is Present at {now.strftime('%H:%M:%S')}")
                else:
                    st.write(f"{name} is already marked present.")

        # Draw rectangles and names on the faces in the image
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.rectangle(img, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(img, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Streamlit app
st.title("Face Recognition Attendance System")

# Initialize WebRTC context with the right processor
try:
    webrtc_ctx = webrtc_streamer(
        key="example",
        video_processor_factory=FaceRecognitionProcessor,
        video_html_attrs={"width": "100%", "height": "100%"},
        media_stream_constraints={"video": True},
        on_error=st.error  # Handle WebRTC errors properly
    )
except Exception as e:
    st.error(f"Error initializing WebRTC: {e}")

# Run the video processing function when 'Start Attendance' button is clicked
if st.button("Start Attendance"):
    st.write("Starting attendance...")
    
    # Wait for the video processor to be initialized
    if webrtc_ctx and webrtc_ctx.video_processor:
        st.write("Waiting for students to appear...")
        
        # Get the list of present students
        present_students = webrtc_ctx.video_processor.present_students
        st.write("Today's Date:", now.strftime("%d-%m-%Y"))

        if present_students:
            st.write("Students who are present today are:")
            for student, time in present_students:
                st.write(f"- {student} - {time}")
    else:
        st.error("WebRTC context initialization failed. Please check your camera.")
else:
    st.write("Click the 'Start Attendance' button to begin.")
