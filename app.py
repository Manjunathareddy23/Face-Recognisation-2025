import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime

# Load known face encodings and names
modi_image = face_recognition.load_image_file("tech-saksham-2025/known_faces/manju.jpeg")
modi_encoding = face_recognition.face_encodings(modi_image)[0]


known_face_encoding = [
    modi_encoding,
    ratan_tata_encoding,
    ujjwal_encoding,
    vivek_encoding,
    sir_encoding
]

known_faces_names = [
    "Manju",
    
]

students = known_faces_names.copy()

# Get the current date for the CSV file
now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

# Open the CSV file for writing attendance
f = open(current_date + '.csv', 'w+', newline='')
lnwriter = csv.writer(f)

class FaceRecognitionProcessor(VideoProcessorBase):
    def __init__(self):
        self.present_students = []

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        small_frame = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        face_names = []

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encoding, face_encoding)
            name = ""
            face_distance = face_recognition.face_distance(known_face_encoding, face_encoding)
            best_match_index = np.argmin(face_distance)
            if matches[best_match_index]:
                name = known_faces_names[best_match_index]

            face_names.append(name)
            if name in known_faces_names:
                if name in students:
                    self.present_students.append((name, now.strftime("%H:%M:%S")))
                    print(name, "is Present")
                    students.remove(name)
                    print("Left Students Name: ", students)
                    current_time = now.strftime("%H-%M-%S")
                    lnwriter.writerow([name, current_time])

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

# Run the video processing function when 'Start Attendance' button is clicked
if st.button("Start Attendance"):
    st.write("Starting attendance...")
    present_students = []
    processor = FaceRecognitionProcessor()
    webrtc_ctx = webrtc_streamer(
        key="example",
        video_processor_factory=FaceRecognitionProcessor,
        async_processing=True,
    )
    if webrtc_ctx.video_processor:
        with st.spinner("Waiting for video..."):
            while not webrtc_ctx.video_processor.present_students:
                pass
            present_students = webrtc_ctx.video_processor.present_students
        st.write("Today's Date:", now.strftime("%d-%m-%Y"))

        if present_students:
            st.write("Students who are present today are:")
            for student, time in present_students:
                st.write(f"- {student} - {time}")
else:
    st.write("Click the 'Start Attendance' button to begin.")
