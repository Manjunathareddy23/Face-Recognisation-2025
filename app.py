import streamlit as st
import cv2
import numpy as np
import pandas as pd
import os
from deepface import DeepFace
from datetime import datetime
import pywhatkit as kit

# Streamlit Configuration
st.set_page_config(page_title='Face Recognition Attendance System', layout='wide')

# Directory and File Paths
FACE_DB_PATH = 'faces/'
ATTENDANCE_DB = 'attendance.csv'

# Ensure directories and files exist
os.makedirs(FACE_DB_PATH, exist_ok=True)

if not os.path.exists(ATTENDANCE_DB):
    pd.DataFrame(columns=['Name', 'Reg_No', 'Date', 'Time']).to_csv(ATTENDANCE_DB, index=False)

# Admin Authentication
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

# Adding New User
def add_new_user():
    st.subheader("Add New User - Capture Face Data")
    name = st.text_input("Student Name:")
    reg_no = st.text_input("Registration Number:")
    contact_no = st.text_input("WhatsApp Contact Number (with country code):")

    if st.button("Capture Face Data"):
        if name and reg_no and contact_no:
            cap = cv2.VideoCapture(0)
            st.info("Align your face with the camera. Press 'q' to capture.")

            while True:
                ret, frame = cap.read()
                if not ret:
                    st.error("Error accessing the camera.")
                    break

                cv2.imshow("Press 'q' to capture", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    file_path = f"{FACE_DB_PATH}/{reg_no}_{name}.jpg"
                    cv2.imwrite(file_path, frame)
                    st.success(f"Face data captured and saved as {file_path}!")
                    break

            cap.release()
            cv2.destroyAllWindows()
        else:
            st.warning("Please fill in all details.")

# Marking Attendance
def mark_attendance(name, reg_no):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    date, time = timestamp.split()
    data = pd.read_csv(ATTENDANCE_DB)

    if not ((data['Name'] == name) & (data['Reg_No'] == reg_no) & (data['Date'] == date)).any():
        new_entry = pd.DataFrame([[name, reg_no, date, time]], columns=data.columns)
        data = pd.concat([data, new_entry], ignore_index=True)
        data.to_csv(ATTENDANCE_DB, index=False)
        st.success(f"Attendance marked for {name} ({reg_no})!")
    else:
        st.warning("Attendance already marked for today!")

# Face Recognition for Attendance
def attendance_page():
    st.subheader("Attendance - Scan Your Face")
    if st.button("Start Camera"):
        cap = cv2.VideoCapture(0)
        known_faces = {file: DeepFace.represent(f"{FACE_DB_PATH}/{file}", model_name='VGG-Face', enforce_detection=False)[0]['embedding']
                       for file in os.listdir(FACE_DB_PATH)}

        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("Error accessing the camera.")
                break

            try:
                result = DeepFace.find(frame, db_path=FACE_DB_PATH, model_name='VGG-Face', enforce_detection=False)
                if not result.empty:
                    file_name = result.iloc[0]['identity'].split('/')[-1]
                    name, reg_no = file_name[:-4].split('_')
                    st.write(f"Welcome {name}! Your Registration Number is {reg_no}.")
                    mark_attendance(name, reg_no)
            except Exception as e:
                st.warning("Face not recognized. Please try again.")

            cv2.imshow("Press 'q' to exit", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

# Student Attendance Report
def view_student_report():
    st.subheader("View Attendance Report")
    reg_no = st.text_input("Enter your Registration Number:")

    if st.button("View Report"):
        data = pd.read_csv(ATTENDANCE_DB)
        student_data = data[data['Reg_No'] == reg_no]

        if not student_data.empty:
            st.dataframe(student_data)
            monthly_data = student_data.groupby(student_data['Date'].str[:7]).size()
            st.bar_chart(monthly_data)
        else:
            st.warning("No attendance record found!")

# Send WhatsApp Notifications
def send_whatsapp_notifications():
    st.subheader("Send Notifications to Absent Students")
    today_date = datetime.now().strftime('%Y-%m-%d')
    data = pd.read_csv(ATTENDANCE_DB)
    absentees = pd.read_csv('students.csv')  # Ensure this CSV exists with ['Name', 'Reg_No', 'Contact_No']

    present_students = data[data['Date'] == today_date]['Reg_No'].unique()
    absent_students = absentees[~absentees['Reg_No'].isin(present_students)]

    if st.button("Send Notifications"):
        for _, student in absent_students.iterrows():
            try:
                kit.sendwhatmsg_instantly(f"+{student['Contact_No']}", 
                                         f"Hi {student['Name']}, you missed today's attendance.", 15)
                st.success(f"Notification sent to {student['Name']} ({student['Contact_No']})!")
            except Exception as e:
                st.error(f"Failed to send message to {student['Name']}")

# Main Application
if admin_login():
    st.sidebar.subheader("Admin Menu")
    options = ["Add New User", "Mark Attendance", "View Attendance Records", "Student Self-Service", "Send Notifications"]
    choice = st.sidebar.radio("Choose an Option:", options)

    if choice == "Add New User":
        add_new_user()
    elif choice == "Mark Attendance":
        attendance_page()
    elif choice == "View Attendance Records":
        st.dataframe(pd.read_csv(ATTENDANCE_DB))
    elif choice == "Student Self-Service":
        view_student_report()
    elif choice == "Send Notifications":
        send_whatsapp_notifications()
else:
    st.warning("Please login to access the admin panel.")
