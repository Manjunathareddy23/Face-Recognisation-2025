# Face Recognition Attendance Web App with Streamlit

import streamlit as st
import cv2
import numpy as np
import face_recognition
import pandas as pd
import os
import datetime
import sqlite3
from PIL import Image
import io

# Database Setup
conn = sqlite3.connect('attendance.db')
cursor = conn.cursor()
cursor.execute('''CREATE TABLE IF NOT EXISTS students (
                    id INTEGER PRIMARY KEY,
                    name TEXT,
                    reg_number TEXT,
                    face_encoding BLOB
                )''')
cursor.execute('''CREATE TABLE IF NOT EXISTS attendance (
                    reg_number TEXT,
                    name TEXT,
                    date DATE,
                    time TIME
                )''')
conn.commit()


# Streamlit UI Design with Tailwind-inspired styling
st.set_page_config(page_title="Face Recognition Attendance System", layout="wide")
st.markdown("<style>body {background-color: #f4f4f9; font-family: Arial, sans-serif;} .main {background-color: #ffffff; padding: 20px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);}</style>", unsafe_allow_html=True)
st.title("🌟 Face Recognition Attendance System")

menu = ["Admin Login", "Attendance"]
choice = st.sidebar.selectbox("🔒 Select Option", menu)

if choice == "Admin Login":
    st.subheader("🔐 Admin Login")
    username = st.text_input("Username", placeholder="Enter admin username")
    password = st.text_input("Password", type="password", placeholder="Enter admin password")

    if st.button("Login"):
        if username == "admin" and password == "password":
            st.success("✅ Logged in as Admin!")
            admin_menu = st.selectbox("Admin Options", ["Add New User", "View Attendance Records", "Generate Reports"])

            if admin_menu == "Add New User":
                st.write("### 📄 Add New User")
                name = st.text_input("Student Name:")
                reg_number = st.text_input("Registration Number:")
                if st.button("Capture Face 📸"):
                    cap = cv2.VideoCapture(0)
                    st.info("Capturing face... Please stay still!")
                    ret, frame = cap.read()
                    cap.release()
                    if ret:
                        st.image(frame)
                        face_encodings = face_recognition.face_encodings(frame)
                        if face_encodings:
                            face_data = np.array(face_encodings[0]).tobytes()
                            cursor.execute("INSERT INTO students (name, reg_number, face_encoding) VALUES (?, ?, ?)", (name, reg_number, face_data))
                            conn.commit()
                            st.success("🎉 Student Registered Successfully!")
                        else:
                            st.error("⚠️ No face detected! Try again.")

            elif admin_menu == "View Attendance Records":
                st.write("### 📊 Attendance Records")
                data = pd.read_sql("SELECT * FROM attendance", conn)
                st.dataframe(data)

            elif admin_menu == "Generate Reports":
                st.write("### 📈 Generate Reports")
                report_type = st.selectbox("Select Report Type", ["Daily", "Monthly", "By Student"])
                if report_type == "Daily":
                    date = st.date_input("Select Date:")
                    data = pd.read_sql(f"SELECT * FROM attendance WHERE date = '{date}'", conn)
                    st.dataframe(data)
                elif report_type == "Monthly":
                    month = st.selectbox("Select Month", range(1, 13))
                    data = pd.read_sql(f"SELECT * FROM attendance WHERE strftime('%m', date) = '{month:02}'", conn)
                    st.dataframe(data)
                elif report_type == "By Student":
                    reg_number = st.text_input("Enter Registration Number:")
                    if st.button("View Attendance"):
                        data = pd.read_sql(f"SELECT * FROM attendance WHERE reg_number = '{reg_number}'", conn)
                        st.dataframe(data)

elif choice == "Attendance":
    st.subheader("📷 Attendance Marking")
    if st.button("Start Camera"):
        cap = cv2.VideoCapture(0)
        st.info("Camera started. Please show your face!")
        ret, frame = cap.read()
        cap.release()
        if ret:
            st.image(frame)
            face_encodings = face_recognition.face_encodings(frame)
            if face_encodings:
                known_faces = cursor.execute("SELECT * FROM students").fetchall()
                match_found = False
                for student in known_faces:
                    db_encoding = np.frombuffer(student[3], dtype=np.float64)
                    matches = face_recognition.compare_faces([db_encoding], face_encodings[0])
                    if True in matches:
                        st.success(f"✅ Attendance Marked for {student[1]} ({student[2]})!")
                        now = datetime.datetime.now()
                        cursor.execute("INSERT INTO attendance (reg_number, name, date, time) VALUES (?, ?, ?, ?)",
                                       (student[2], student[1], now.date(), now.time()))
                        conn.commit()
                        match_found = True
                        break
                if not match_found:
                    st.warning("❗ Unrecognized Face! Please Contact Admin.")
            else:
                st.error("⚠️ No face detected! Try again.")


conn.close()
