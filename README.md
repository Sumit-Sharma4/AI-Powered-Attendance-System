# AI-Powered Attendance System ✅
**Face Recognition Based Smart Attendance System with Anti-Spoofing & Liveness Detection**

This project is an **AI-powered attendance system** that marks attendance using **real-time face recognition** with **anti-spoofing protection** and **blink-based liveness detection** to prevent fake attendance using photos/videos.

> 🎓 Academic Project (B.Tech CSE)  
> 🔐 Focus: Secure & Smart Attendance Automation

---

## 🚀 Key Features

✅ **Real-time Face Detection & Recognition**  
✅ **Blink-based Liveness Detection (MediaPipe)**  
✅ **Anti-Spoofing Model Integration (MiniFASNet)**  
✅ **Student & Faculty Registration Module**  
✅ **Automatic Attendance Marking (CSV)**  
✅ **Unique Attendance Entry (No duplicate attendance)**  
✅ **Email Notification Support (Optional Module)**  
✅ Clean & Modular Code Structure (src-based)

---

## 🧠 Technology Stack

- **Python 3.x**
- **OpenCV** (Webcam + Face Detection)
- **DeepFace** (Face Recognition)
- **MediaPipe** (Blink/Liveness Detection)
- **PyTorch** (Anti-Spoofing Model)
- **NumPy / Pandas** (Data handling)
- **CSV** (Attendance storage)

## requirements.txt
requests>=2.27.1
numpy>=1.14.0
pandas>=0.23.4
gdown>=3.10.1
tqdm>=4.30.0
Pillow>=5.2.0
opencv-python>=4.5.5.64
tensorflow>=2.9.0
keras>=2.9.0
Flask>=1.1.2
flask_cors>=4.0.1
mtcnn>=0.1.0
retina-face>=0.0.1
fire>=0.4.0
gunicorn>=20.1.0

---

## 📂 Project Structure

```bash
AI-Powered-Attendance-System/
│── VAttendance.py
│── README.md
│── requirements.txt
│── .gitignore
│
├── resources/
│   ├── detection_model/
│   │   ├── deploy.prototxt
│   │   └── Widerface-RetinaFace.caffemodel
│   └── anti_spoof_models/   # (Not uploaded on GitHub - large models)
│
├── src/
│   ├── anti_spoof_predict.py
│   ├── default_config.py
│   ├── generate_patches.py
│   ├── utility.py
│   ├── data_io/
│   └── model_lib/
│
├── TrainingImage/           # (Ignored for privacy)
├── TrainingImageLabel/      # (Ignored for privacy)
└── Attendance/              # (Ignored for privacy)
 
