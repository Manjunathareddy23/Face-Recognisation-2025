import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import os
import face_recognition
import mediapipe as mp
from datetime import datetime
import numpy as np
import random
import pandas as pd

# Global Variables
is_capturing = False
cap = None
image_count = 0
max_images = 10
folder = ""
encoded_faces = []
student_names = []
EYE_AR_THRESH = 0.25  
EYE_AR_CONSEC_FRAMES = 8 
required_blinks = 0
detected_blinks = 0
fps = 4
is_check = False  
label2 = ""
right_label = ""
attendance_window = None

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=10)

def set_random_blink_requirement():
    global required_blinks, detected_blinks
    required_blinks = random.randint(1, 2) 
    detected_blinks = 0

def eye_aspect_ratio(eye_landmarks):
    A = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
    B = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
    C = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
    ear = (A + B) / (2.0 * C)
    return ear

def blink_detected(landmarks):
    left_eye = [landmarks[i] for i in [33, 160, 158, 133, 153, 144]]
    right_eye = [landmarks[i] for i in [362, 385, 387, 263, 373, 380]]
    left_ear = eye_aspect_ratio(np.array(left_eye))
    right_ear = eye_aspect_ratio(np.array(right_eye))
    ear = (left_ear + right_ear) / 2.0
    return ear

def start_capture():
    global is_capturing, cap, folder, progress_var
    student_name = entry_name.get()
    if not student_name:
        messagebox.showwarning("Peringatan", "Masukkan nama mahasiswa terlebih dahulu!")
        return

    folder = f"dataset/{student_name}"
    os.makedirs(folder, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Error", "Kamera tidak dapat dibuka!")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    is_capturing = True
    messagebox.showinfo("Info", "Mulai merekam gambar. Total 10 gambar akan diambil.")

    progress_var.set(0)
    progress_bar['maximum'] = max_images
    capture_images()

def capture_images():
    global is_capturing, image_count, max_images, cap, progress_var
    if not is_capturing:
        return

    ret, frame = cap.read()
    if not ret:
        messagebox.showerror("Error", "Gagal membaca frame dari kamera!")
        return

    face_locations = face_recognition.face_locations(frame)
    if len(face_locations) > 0:  
        for (top, right, bottom, left) in face_locations:
            face_image = frame[top:bottom, left:right]

            img_name = f"{folder}/{entry_name.get()}_{image_count + 1}.jpg"
            cv2.imwrite(img_name, face_image)

            image_count += 1
            progress_var.set(image_count)
            percentage = (image_count / max_images) * 100
            percentage_label.config(text=f"{percentage:.0f}%")
            if image_count < max_images:
                break  #
    else:
        print("Tidak ada wajah yang terdeteksi.")

    if image_count < max_images:
        root.after(1000, capture_images)
    else:
        cap.release()
        encode_faces()
        messagebox.showinfo("Selesai", f"Pengambilan gambar selesai. {max_images} gambar telah diambil dan encoding wajah selesai!")
        is_capturing = False

def encode_faces():
    global encoded_faces, student_names, image_count

    if not os.path.exists('dataset'):
        messagebox.showwarning("Peringatan", "Dataset tidak ditemukan! Silakan ambil gambar terlebih dahulu.")
        return

    for student_folder in os.listdir('dataset'):
        folder_path = f'dataset/{student_folder}'
        if os.path.isdir(folder_path):
            for image_file in os.listdir(folder_path):
                img_path = f'{folder_path}/{image_file}'
                img = face_recognition.load_image_file(img_path)
                face_encodings = face_recognition.face_encodings(img)

                if face_encodings:
                    encoded_faces.append(face_encodings[0])
                    student_names.append(student_folder)

    progress_var.set(0)
    image_count = 0
    percentage_label.config(text="0%") 

def start_attendance():
    global encoded_faces, student_names, detected_blinks, required_blinks

    set_random_blink_requirement() 
    encode_faces()  

    if not encoded_faces:
        messagebox.showwarning("Peringatan", "Encode wajah terlebih dahulu!")
        return

    cap = cv2.VideoCapture(0) 
    if not cap.isOpened():
        messagebox.showerror("Error", "Kamera tidak dapat dibuka!")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    attendance_list = set()  
    attendance_status = {}  

    def mark_attendance(name):
        global attendance_window
        now = datetime.now()
        timestamp = now.strftime('%Y-%m-%d %H:%M:%S')
        if name not in attendance_list and name != "Unknown":
            attendance_list.add(name)
            attendance_status[name] = "(DONE)" 

            with open('attendance.csv', 'a') as f:
                f.write(f'{name},{timestamp}\n')

            root.after(100, check_attendance)  

            print(f"{name} telah diabsen pada {timestamp}")

    blink_frame_counters = {} 
    detected_blinks_per_face = {}
    blink_detected_flags = {}  
    blink_time = EYE_AR_CONSEC_FRAMES / fps 

    while True:
        ret, frame = cap.read()
        if not ret:
            messagebox.showerror("Error", "Gagal membaca frame dari kamera!")
            break

        frame = cv2.resize(frame, (640, 480))
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = face_mesh.process(rgb_frame)
        if results.multi_face_landmarks:
            face_landmarks_list = results.multi_face_landmarks

            for i, face_landmarks in enumerate(face_landmarks_list):
                landmarks = [(landmark.x * frame.shape[1], landmark.y * frame.shape[0]) for landmark in face_landmarks.landmark]

                ear = blink_detected(landmarks)

                face_locations = face_recognition.face_locations(rgb_frame)
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

                for face_encoding, face_location in zip(face_encodings, face_locations):
                    matches = face_recognition.compare_faces(encoded_faces, face_encoding)
                    face_distances = face_recognition.face_distance(encoded_faces, face_encoding)
                    best_match_index = np.argmin(face_distances)

                    if matches[best_match_index] and face_distances[best_match_index] < 0.4:
                        name = student_names[best_match_index]
                    else:
                        name = "Unknown"

                    if name not in detected_blinks_per_face:
                        detected_blinks_per_face[name] = 0

                    if ear < EYE_AR_THRESH:
                        if name in blink_frame_counters:
                            blink_frame_counters[name] += 1
                        else:
                            blink_frame_counters[name] = 1
                    else:
                        if name in blink_frame_counters and blink_frame_counters[name] >= EYE_AR_CONSEC_FRAMES:
                            detected_blinks_per_face[name] += 1  
                            blink_frame_counters[name] = 0 

                    if detected_blinks_per_face[name] >= required_blinks:
                        print(f"\033[92mJumlah kedipan untuk {name} terpenuhi!\033[0m")
                        blink_detected_flags[name] = True
                        detected_blinks_per_face[name] = 0  

                    if blink_detected_flags.get(name, False) and name != "Unknown":
                        mark_attendance(name)
                        blink_detected_flags[name] = False  

                    top, right, bottom, left = face_location
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2) 

                    blink_count = detected_blinks_per_face.get(name, 0)
                    status = attendance_status.get(name, "")  
                    if status == "(DONE)":
                        label = f"{name} {status}"  
                    else:
                        blink_count = detected_blinks_per_face.get(name, 0)
                        if name == "Unknown": 
                             label = f"{name}"  
                        else:
                             label = f"{name} ({blink_count})"  

                    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    label_y_min = max(top, label_size[1] + 10)
                    cv2.rectangle(frame, (left, label_y_min - label_size[1] - 10), 
                                  (left + label_size[0], label_y_min + 5), (0, 255, 0), cv2.FILLED)  
                    cv2.putText(frame, label, (left, label_y_min - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)  

        cv2.putText(frame, f"Jumlah kedip: {required_blinks}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Waktu kedip: {blink_time}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow('Absensi Wajah', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
            break

    cap.release()
    cv2.destroyAllWindows()
