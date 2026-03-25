import dlib
import numpy as np
import cv2
import os
import pandas as pd
import time
import logging
import sqlite3
import datetime
from scipy.spatial import distance as dist
from imutils import face_utils

# Load dlib detector & models
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('data/data_dlib/shape_predictor_68_face_landmarks.dat')
face_reco_model = dlib.face_recognition_model_v1("data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")

# Create database table if not exists
conn = sqlite3.connect("attendance.db")
cursor = conn.cursor()
cursor.execute("CREATE TABLE IF NOT EXISTS attendance (name TEXT, time TEXT, date DATE, UNIQUE(name, date))")
conn.commit()
conn.close()

# EAR (eye aspect ratio)
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)


class Face_Recognizer:
    def __init__(self):
        self.font = cv2.FONT_HERSHEY_SIMPLEX

        # face database
        self.face_features_known_list = []
        self.face_name_known_list = []

        # blink detection settings
        self.EAR_THRESHOLD = 0.21
        self.EAR_CONSEC_FRAMES = 3

        self.blink_counter = {}  # count closed-eye frames
        self.total_blinks = {}   # total blinks
        self.liveness_ok = {}    # true if blink twice

    # Load known faces
    def get_face_database(self):
        path = "data/features_all.csv"
        if not os.path.exists(path):
            print("[ERROR] features_all.csv NOT FOUND!")
            return False

        csv_rd = pd.read_csv(path, header=None)
        for i in range(len(csv_rd)):
            name = csv_rd.iloc[i][0]
            features = csv_rd.iloc[i][1:].astype(float).tolist()

            self.face_name_known_list.append(name)
            self.face_features_known_list.append(features)

        print("[INFO] Loaded", len(self.face_name_known_list), "registered faces.")
        return True

    # Mark attendance
    def attendance(self, name):
        today = datetime.datetime.now().strftime('%Y-%m-%d')

        conn = sqlite3.connect("attendance.db")
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM attendance WHERE name=? AND date=?", (name, today))
        if cursor.fetchone():
            print(f"[ATT] Already marked today: {name}")
            conn.close()
            return

        now = datetime.datetime.now().strftime('%H:%M:%S')
        cursor.execute("INSERT INTO attendance (name, time, date) VALUES (?, ?, ?)", (name, now, today))
        conn.commit()
        conn.close()

        print(f"[ATT] Attendance Marked ➤ {name} at {now}")

    # Main loop
    def process(self, stream):
        if not self.get_face_database():
            return

        while stream.isOpened():
            ret, frame = stream.read()
            if not ret:
                break

            gray = frame
            faces = detector(gray, 0)

            for i, face in enumerate(faces):

                shape = predictor(gray, face)
                shape_np = face_utils.shape_to_np(shape)

                # ======= DRAW FACE BOX =======
                cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (255, 255, 255), 2)

                # EAR detection
                left_eye = shape_np[36:42]
                right_eye = shape_np[42:48]
                ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0

                if i not in self.blink_counter:
                    self.blink_counter[i] = 0
                    self.total_blinks[i] = 0
                    self.liveness_ok[i] = False

                if ear < self.EAR_THRESHOLD:
                    self.blink_counter[i] += 1
                else:
                    if self.blink_counter[i] >= self.EAR_CONSEC_FRAMES:
                        self.total_blinks[i] += 1
                        print(f"Face {i} blinked ({self.total_blinks[i]})")
                    self.blink_counter[i] = 0

                if self.total_blinks[i] >= 2:
                    self.liveness_ok[i] = True

                # ======== FACE RECOGNITION ========
                descriptor = face_reco_model.compute_face_descriptor(gray, shape)
                descriptor = np.array(descriptor)

                distances = [
                    np.linalg.norm(descriptor - np.array(known))
                    for known in self.face_features_known_list
                ]

                min_dist = min(distances)
                idx = distances.index(min_dist)

                if min_dist < 0.40:
                    name = self.face_name_known_list[idx]
                else:
                    name = "Unknown"

                # ======== DISPLAY NAME ========
                cv2.putText(frame, name, (face.left(), face.top() - 10),
                            self.font, 0.8, (0, 255, 255), 2)

                # Only registered names can mark attendance
                if name != "Unknown":
                    if self.liveness_ok[i]:
                        self.attendance(name)
                        self.total_blinks[i] = 0
                        self.liveness_ok[i] = False
                    else:
                        cv2.putText(frame, "Blink twice to verify!",
                                    (face.left(), face.bottom() + 20),
                                    self.font, 0.6, (0, 0, 255), 2)

            cv2.imshow("Attendance Camera", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def run(self):
        cap = cv2.VideoCapture(0)
        self.process(cap)
        cap.release()
        cv2.destroyAllWindows()


def main():
    Face_Recognizer().run()


if __name__ == "__main__":
    main()
