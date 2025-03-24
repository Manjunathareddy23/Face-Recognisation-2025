# Face Recognition Attendance System using Streamlit

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import os
import face_recognition
from datetime import datetime
import pywhatkit as kit  # For WhatsApp notifications

# Streamlit Page Configurations
st.set_page_config(page_title='Face Recognition Attendance System', layout='wide')

# Database paths
FACE_DB_PATH = 'faces/'
ATTENDANCE_DB = 'attendance.csv'

if not os.path.exists(FACE_DB_PATH):
    os.makedirs(FACE_DB_PATH)

if not os.path.exists(ATTENDANCE_DB):
    pd.DataFrame(columns=['Name', 'Reg_No', 'Date', 'Time']).to_csv(ATTENDANCE_DB, index=False)


# Authentication for Admin
def admin_login():
    st.sidebar.title("Admin Login")
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")
    if st.sidebar.button("Login"):
        if username == 'admin' and password == 'admin123':
            st.success("Login Successful!")
            return True
        else:
            st.error("Invalid Credentials!")
    return False


# Capture New Student Data
def add_new_user():
    st.subheader("Add New User - Capture Face Data")
    name = st.text_input("Student Name:")
    reg_no = st.text_input("Registration Number:")
    contact_no = st.text_input("WhatsApp Contact Number (with country code):")

    if st.button("Capture Face Data"):
        if name and reg_no and contact_no:
            cap = cv2.VideoCapture(0)
            st.info("Capturing face data. Please align your face with the camera.")

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                cv2.imshow("Capture - Press 'q' to save", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    face_encodings = face_recognition.face_encodings(frame)
                    if face_encodings:
                        np.save(f"{FACE_DB_PATH}/{reg_no}_{name}.npy", face_encodings[0])
                        st.success("Face data captured and saved successfully!")
                        break
                    else:
                        st.warning("No face detected. Please try again.")
            cap.release()
            cv2.destroyAllWindows()
        else:
            st.warning("Please fill in all details.")


# Mark Attendance
def mark_attendance(name, reg_no):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    date, time = timestamp.split()
    data = pd.read_csv(ATTENDANCE_DB)

    if not ((data['Name'] == name) & (data['Reg_No'] == reg_no) & (data['Date'] == date)).any():
        data = pd.concat([data, pd.DataFrame([[name, reg_no, date, time]], columns=data.columns)], ignore_index=True)
        data.to_csv(ATTENDANCE_DB, index=False)
        st.success(f"Attendance marked for {name} ({reg_no})!")
    else:
        st.warning("Attendance already marked for today!")


# Face Recognition - Attendance
def attendance_page():
    st.subheader("Attendance - Scan Your Face")

    if st.button("Start Camera"):
        cap = cv2.VideoCapture(0)
        known_faces = []
        known_names = []

        for file in os.listdir(FACE_DB_PATH):
            face_data = np.load(FACE_DB_PATH + file)
            known_faces.append(face_data)
            known_names.append(" ".join(file[:-4].split('_')[1:]))

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            face_locations = face_recognition.face_locations(frame)
            face_encodings = face_recognition.face_encodings(frame, face_locations)

            for encoding in face_encodings:
                matches = face_recognition.compare_faces(known_faces, encoding)
                if True in matches:
                    matched_index = matches.index(True)
                    name, reg_no = known_names[matched_index].rsplit(' ', 1)
                    st.write(f"Welcome {name}! Your Registration Number is {reg_no}.")
                    mark_attendance(name, reg_no)

            cv2.imshow("Press 'q' to exit", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


# Student Self-Service for Attendance Report
def view_student_report():
    st.subheader("Student Self-Service - View Attendance Report")
    reg_no = st.text_input("Enter your Registration Number:")
    
    if st.button("View Report"):
        data = pd.read_csv(ATTENDANCE_DB)
        student_data = data[data['Reg_No'] == reg_no]

        if not student_data.empty:
            st.dataframe(student_data)
            monthly_data = student_data.groupby(student_data['Date'].str[:7]).size()
            st.write("Monthly Attendance Summary:")
            st.bar_chart(monthly_data)
        else:
            st.warning("No attendance record found!")


# Send WhatsApp Notifications to Absent Students
def send_whatsapp_notifications():
    st.subheader("Send Notifications to Absent Students")
    today_date = datetime.now().strftime('%Y-%m-%d')
    data = pd.read_csv(ATTENDANCE_DB)

    absentees = pd.read_csv('students.csv')  # Assuming a student contact list exists
    present_students = data[data['Date'] == today_date]['Reg_No'].unique()
    absent_students = absentees[~absentees['Reg_No'].isin(present_students)]

    if st.button("Send Notifications"):
        for _, student in absent_students.iterrows():
            contact_no = student['Contact_No']
            name = student['Name']
            kit.sendwhatmsg_instantly(f"+{contact_no}", f"Hi {name}, you missed today's attendance. Please report to the admin.", 10)
            st.success(f"Notification sent to {name} ({contact_no})!")


# Main Application
if admin_login():
    st.sidebar.subheader("Admin Menu")
    choice = st.sidebar.radio("Choose an Option:", ["Add New User", "Mark Attendance", "View Attendance Records", "Student Self-Service", "Send Notifications"])

    if choice == "Add New User":
        add_new_user()
    elif choice == "Mark Attendance":
        attendance_page()
    elif choice == "View Attendance Records":
        data = pd.read_csv(ATTENDANCE_DB)
        st.dataframe(data)
        st.download_button("Download CSV", data.to_csv(index=False), "attendance.csv")
    elif choice == "Student Self-Service":
        view_student_report()
    elif choice == "Send Notifications":
        send_whatsapp_notifications()
else:
    st.warning("Please login to access the admin panel.")
