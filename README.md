# Smart Attendance Monitoring Through Facial Biometrics

### With Blink-Based Liveness Detection & Anti-Spoofing

A real-time face recognition-based attendance system built using OpenCV, Dlib, Deep Learning, and Blink Detection to ensure secure and fraud-free attendance marking. Supports multiple persons simultaneously, live database updates, and a clean web dashboard.

---

## 🚀 Features

### 🔹 Automated Face Recognition

Uses Dlib’s ResNet (128D face embeddings) for accurate recognition in various conditions.

### 🔹 Blink-Based Liveness Detection

Prevents fraud such as printing a photo or showing a video. Attendance is marked only after **two successful blinks**.

### 🔹 Multi-Face Detection

Each face has:

* Independent EAR (eye aspect ratio)
* Blink counter
* Liveness validation

### 🔹 Attendance Marking

* Only registered faces can mark attendance
* Unknown faces are shown but ignored
* Attendance stored with **Name, Time, Date** in SQLite DB

### 🔹 Web Dashboard (Flask)

* Select a date to view attendance
* Clean interface using HTML + Jinja2

---

## 🧠 Tech Stack

| Component          | Technology          |
| ------------------ | ------------------- |
| Language           | Python              |
| Computer Vision    | OpenCV              |
| Face Detection     | Dlib HOG / CNN      |
| Face Recognition   | Dlib ResNet-50      |
| Liveness Detection | EAR Blink Detection |
| Web Framework      | Flask               |
| Database           | SQLite              |

---

## 📁 Folder Structure

```
Smart-Attendance-Monitoring-Through-Facial-Biometrics/
│
├── attendance_taker.py
├── get_faces_from_camera_tkinter.py
├── features_extraction_to_csv.py
├── app.py
├── attendance.db
│
├── data/
│   ├── data_faces_from_camera/
│   ├── data_dlib/
│   │     ├── shape_predictor_68_face_landmarks.dat
│   │     └── dlib_face_recognition_resnet_model_v1.dat
│   └── features_all.csv
│
└── templates/
    └── index.html
```

---

## 🛠 Installation

### 1. Clone the repository

```
git clone https://github.com/upendra-tata/Smart-attendance-monitoring-through-facial-biometrics.git
cd Smart-attendance-monitoring-through-facial-biometrics
```

### 2. Create virtual environment

```
python -m venv .venv
```

### 3. Activate environment

```
.venv\Scripts\activate
```

### 4. Install required packages

```
pip install -r requirements.txt
```

If missing:

```
pip install opencv-python dlib imutils scipy numpy flask pandas
```

---

## 🎥 How to Use

### ✔ Register Your Face

```
python get_faces_from_camera_tkinter.py
```

### ✔ Generate Face Embeddings

```
python features_extraction_to_csv.py
```

### ✔ Start Attendance System

```
python attendance_taker.py
```

The system will:

* Detect faces
* Display Name or Unknown
* Request 2 blinks for verification
* Mark attendance after validation

### ✔ View Attendance Dashboard

```
python app.py
```

Open browser:

```
http://127.0.0.1:5000/
```

---

## 🔒 Security

* Blink validation prevents spoof attacks
* Per-face liveness tracking for multi-person detection

---

## 📈 Future Enhancements

* Head pose or mouth-movement detection
* Export attendance to Excel
* Admin login
* Real-time camera feed on web
* Cloud deployment support

---

## 🤝 Contributing

Pull requests are welcome. Open an issue for feature requests.

---

## 📜 License

Project is licensed under the MIT License.

---

## 🙌 Credits

Developed by using Dlib, OpenCV, and Flask.

