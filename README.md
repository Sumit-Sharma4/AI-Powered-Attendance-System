# AI-Powered Attendance System ✅
**Face Recognition Based Smart Attendance System with Anti-Spoofing & Liveness Detection**

This project is an **AI-powered attendance system** that marks attendance using **real-time face recognition** with **anti-spoofing protection** and **blink-based liveness detection** to prevent fake attendance using photos/videos.
  
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
```
---

## 📌 How It Works (Workflow)

### ✅ Step 1: Registration
- Enter **Student/Faculty details**
- Capture images using **webcam**
- Store data in **CSV file** and **training folder**

### ✅ Step 2: Liveness Verification
- Uses **blink detection** to ensure the face is real (not a photo)

### ✅ Step 3: Anti-Spoofing Check
- Anti-spoof model detects spoof attempts such as:
  - Printed photo
  - Mobile screen photo/video

### ✅ Step 4: Mark Attendance
- If Face is verified + Real (live) → Attendance is marked
- Attendance is saved automatically in a **CSV file**

---

## 🧪 Output
- Live webcam **face detection & recognition**
- Liveness verification using **blink detection**
- Attendance saved automatically into a **CSV file**

---

  ## 📸 Screenshots

### ✅ Home Screen
![Home Screen](<img width="1919" height="1025" alt="Screenshot 2025-12-15 103125" src="https://github.com/user-attachments/assets/194f83ec-56ca-49c0-a274-f266a3d9dc37" />
)

### ✅ Student Registration
![Student Registration](<img width="376" height="342" alt="Screenshot 2025-12-15 103149" src="https://github.com/user-attachments/assets/8b4556b9-f147-465c-8e3a-46e2519cc94b" />
)

### ✅ Attendance Marking
![Attendance](![SC](https://github.com/user-attachments/assets/cec283b4-013b-41f9-9229-e48162d3a512)
)

---

## ✨ Future Enhancements
✅ Web Dashboard (Flask/Django) for admin  
✅ Cloud Database Integration (MongoDB/MySQL)  
✅ QR + Face Hybrid Attendance  
✅ Better UI & report generation (PDF export)  
✅ Deployable exe setup using PyInstaller  

---

## 👨‍💻 Author
**Sumit Sharma**  

