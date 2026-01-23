# VAttendance.py
# DeepFace + MediaPipe blink + SilentFace Anti-spoof + Email + TTS name announce

import tkinter as tk
from tkinter import ttk, messagebox, simpledialog, BooleanVar, Checkbutton
import cv2, os, numpy as np
import pandas as pd
import time
from datetime import datetime, timedelta
from collections import deque, defaultdict
import math
import hashlib
import smtplib
import ssl
from email.message import EmailMessage
import threading

# ---------- NEW: Text-to-Speech ----------
try:
    import pyttsx3
    TTS_AVAILABLE = True
    # we'll use per-call engine in speak_name_async
except Exception:
    TTS_AVAILABLE = False

def speak(text: str):
    """
    (Legacy, not used now) - kept for compatibility.
    We now use speak_name_async instead.
    """
    if not TTS_AVAILABLE:
        return
    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", 170)
        engine.setProperty("volume", 1.0)
        engine.say(text)
        engine.runAndWait()
        engine.stop()
    except Exception:
        pass

def speak_name_async(text):
    """
    Speak given text in a background thread using pyttsx3.
    New engine per call to avoid 'only first time speaks' issue.
    """
    if not TTS_AVAILABLE:
        return

    def worker():
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 175)    # speed
            engine.setProperty('volume', 1.0)  # full volume
            engine.say(text)
            engine.runAndWait()
            engine.stop()
        except Exception as e:
            print("TTS error:", e)

    threading.Thread(target=worker, daemon=True).start()


# DeepFace (FaceNet / SFace)
from deepface import DeepFace

# Mediapipe for blink detection
try:
    import mediapipe as mp
    MP_AVAILABLE = True
except Exception:
    MP_AVAILABLE = False

# ---------------- REAL ANTI-SPOOF (SilentFace / PyTorch) ----------------
# Clone Silent-Face-Anti-Spoofing repo so that this file sits next to /src and /resources
# https://github.com/minivision-ai/Silent-Face-Anti-Spoofing
try:
    # Force CPU (optional – remove if you want GPU)
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    from src.anti_spoof_predict import AntiSpoofPredict
    from src.generate_patches import CropImage
    from src.utility import parse_model_name

    SILENTFACE_AVAILABLE = True
except Exception as e:
    print("[SilentFace] Import error:", e)
    SILENTFACE_AVAILABLE = False

# In the official repo, anti-spoof models are under resources/anti_spoof_models
SILENTFACE_MODEL_DIR = os.path.join("resources", "anti_spoof_models")
SILENTFACE_THRESHOLD = 0.7   # LIVE threshold for current frame


class SilentFaceLiveness:
    """
    Thin wrapper around Silent-Face-Anti-Spoofing for live score in [0,1].
    label: 1 = live, 0/2 = spoof/other
    """

    def __init__(self, device_id=0):
        self.device_id = device_id
        self.model = None
        self.cropper = None
        self.model_files = []

        if not SILENTFACE_AVAILABLE:
            print("[SilentFace] Not available (imports failed).")
            return

        try:
            self.model = AntiSpoofPredict(device_id)
            self.cropper = CropImage()

            if not os.path.isdir(SILENTFACE_MODEL_DIR):
                print("[SilentFace] Model dir not found:", SILENTFACE_MODEL_DIR)
            else:
                self.model_files = [
                    os.path.join(SILENTFACE_MODEL_DIR, f)
                    for f in os.listdir(SILENTFACE_MODEL_DIR)
                    if f.lower().endswith(".pth")
                ]
                if not self.model_files:
                    print("[SilentFace] No .pth models found in", SILENTFACE_MODEL_DIR)
                else:
                    print("[SilentFace] Loaded model files:", len(self.model_files))
        except Exception as e:
            print("[SilentFace] Init error:", e)
            self.model = None
            self.cropper = None
            self.model_files = []

    def score(self, frame_bgr, bbox_xywh):
        """
        frame_bgr : full frame (H,W,3) BGR from OpenCV
        bbox_xywh : [x, y, w, h] from your Haar detector
        returns (label, live_score)

        label: 1 => real, other => spoof
        live_score: 0..1 (higher => more real)
        """
        if self.model is None or self.cropper is None or not self.model_files:
            return 0, 0.0

        x, y, w, h = map(int, bbox_xywh)
        h_img, w_img = frame_bgr.shape[:2]

        # Clamp bbox to image
        x = max(0, min(x, w_img - 1))
        y = max(0, min(y, h_img - 1))
        w = max(1, min(w, w_img - x))
        h = max(1, min(h, h_img - y))

        image_bbox = [x, y, w, h]
        prediction = np.zeros((1, 3), dtype="float32")

        for model_path in self.model_files:
            model_name = os.path.basename(model_path)
            try:
                h_input, w_input, model_type, scale = parse_model_name(model_name)
            except Exception:
                # If parse fails, skip this model
                continue

            param = {
                "org_img": frame_bgr,
                "bbox": image_bbox,
                "scale": scale,
                "out_w": w_input,
                "out_h": h_input,
                "crop": True,
            }
            if scale is None:
                param["crop"] = False

            try:
                crop_img = self.cropper.crop(**param)
            except Exception:
                crop_img = None

            if crop_img is None:
                continue

            try:
                # returns (1,3): [fake, real, other]
                pred = self.model.predict(crop_img, model_path)
                prediction += pred
            except Exception:
                continue

        if not np.any(prediction):
            return 0, 0.0

        label = int(np.argmax(prediction))
        fake_score, real_score, other_score = prediction[0]

        total = float(fake_score + real_score + other_score + 1e-8)
        live_score = float(real_score) / total
        live_score = max(0.0, min(1.0, live_score))

        return label, live_score


# Create global liveness object
silentface_liveness = SilentFaceLiveness(device_id=0)

# ---------------- CONFIG & PATHS ----------------
ADMIN_PASSWORD = "1234"
CASCADE_PATH = "haarcascade_frontalface_default.xml"
if not os.path.exists(CASCADE_PATH):
    CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

STUDENT_CSV = "StudentDetails.csv"         # columns: Id, Name
FACULTY_CSV = "FacultyDetails.csv"         # columns: Id, Name, Dept, Subject, PasswordHash, Email
IMAGE_DIR   = "TrainingImage"              # registration pictures live here: Name.Id.#.jpg
ATTENDANCE_DIR = "Attendance"

os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(ATTENDANCE_DIR, exist_ok=True)

# ---------------- RUNTIME PARAMS ----------------
# Distance/scale gate (applies to registration & attendance)
FACE_MIN_SIZE = 100       # face bbox width (px) -> too small = too far
FACE_MAX_SIZE = 270       # face bbox width (px) -> too large = too near

# Sharpness (blurry face skip)
SHARPNESS_THRESHOLD = 80.0  # Laplacian variance

# NEW: Brightness threshold for avoiding low-light spoof misclassification
BRIGHTNESS_MIN = 60.0    # you can tune (50–80). Lower = allows darker faces.

# Blink / EAR params
EAR_THRESHOLD = 0.27
EAR_CONSEC_FRAMES = 2
LEFT_EYE_IDX  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

# Tracking & logic
FRAME_THRESHOLD = 2          # frames of stable prediction to accept recognition
MATCH_DISTANCE_PX = 80       # centroid re-assignment distance
TRACK_INACTIVE_FRAMES = 30   # prune tracks after N frames

# DeepFace model & threshold (SFace faster on CPU)
DF_MODEL_NAME = "SFace"
FACENET_COSINE_THRESHOLD = 0.5  # smaller -> stricter

# Beep cooldown (seconds) for “Perfect distance” (we no longer beep, but kept if needed)
BEEP_COOLDOWN_SEC = 15.0

# ---------------- HELPERS ----------------
def now_date_str(): return datetime.now().strftime("%Y-%m-%d")
def now_time_str(): return datetime.now().strftime("%H:%M:%S")
def sha256_hash(text): return hashlib.sha256(text.encode("utf-8")).hexdigest()
def euclidean(a, b): return math.hypot(a[0]-b[0], a[1]-b[1])

def is_face_sharp(gray_img):
    try:
        return cv2.Laplacian(gray_img, cv2.CV_64F).var() > SHARPNESS_THRESHOLD
    except Exception:
        return False

def load_students_df():
    if os.path.exists(STUDENT_CSV):
        return pd.read_csv(STUDENT_CSV, dtype={"Id": str, "Name": str})
    return pd.DataFrame(columns=["Id", "Name"], dtype=str)

def save_students_df(df):
    df.to_csv(STUDENT_CSV, index=False)

def load_faculty_df():
    cols = ["Id", "Name", "Dept", "Subject", "PasswordHash", "Email"]
    if os.path.exists(FACULTY_CSV):
        df = pd.read_csv(FACULTY_CSV, dtype=str)
        for c in cols:
            if c not in df.columns:
                df[c] = ""
        return df[cols].fillna("")
    df = pd.DataFrame(columns=cols, dtype=str)
    return df

def save_faculty_df(df):
    cols = ["Id", "Name", "Dept", "Subject", "PasswordHash", "Email"]
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    df[cols].to_csv(FACULTY_CSV, index=False)

def attendance_file_path(date_str):
    return os.path.join(ATTENDANCE_DIR, f"Attendance_{date_str}.csv")

def load_today_attendance(date_str):
    path = attendance_file_path(date_str)
    cols = ["FacultyId", "FacultyName", "FacultyDept", "FacultySubject",
            "StudentId", "StudentName", "Date", "Time"]
    if os.path.exists(path):
        df = pd.read_csv(path, dtype=str)
        for c in cols:
            if c not in df.columns:
                df[c] = ""
        return df[cols]
    return pd.DataFrame(columns=cols, dtype=str)

def append_to_today_attendance(row):
    """
    row = [FacultyId, FacultyName, FacultyDept, FacultySubject,
           StudentId, StudentName, Date, Time]
    Dedup by (FacultyId, FacultySubject, StudentId)
    """
    date_str = now_date_str()
    path = attendance_file_path(date_str)
    cols = ["FacultyId", "FacultyName", "FacultyDept", "FacultySubject",
            "StudentId", "StudentName", "Date", "Time"]
    new_entry = pd.DataFrame([row], columns=cols)
    if os.path.exists(path):
        df_old = pd.read_csv(path, dtype=str)
        for c in cols:
            if c not in df_old.columns:
                df_old[c] = ""
        df_old = df_old[cols]
        df = pd.concat([df_old, new_entry], ignore_index=True)
        df.drop_duplicates(subset=["FacultyId", "FacultySubject", "StudentId"],
                           keep="first", inplace=True)
        df.to_csv(path, index=False)
    else:
        new_entry.to_csv(path, index=False)

# EAR from landmarks
def compute_ear_from_landmarks(lm_coords, eye_indices):
    def dist(a, b): return math.hypot(a[0]-b[0], a[1]-b[1])
    try:
        p = [lm_coords[i] for i in eye_indices]
    except Exception:
        return 0.0
    A = dist(p[1], p[5]); B = dist(p[2], p[4]); C = dist(p[0], p[3])
    if C == 0: return 0.0
    return (A + B) / (2.0 * C)

def compute_ear_safe(lm_coords, eye_indices):
    try:
        if not lm_coords or len(lm_coords) <= max(eye_indices):
            return 0.0
    except Exception:
        return 0.0
    return compute_ear_from_landmarks(lm_coords, eye_indices)

# ---------------- DeepFace embedding cache ----------------
student_embeddings = {}

def parse_id_from_filename(fname_no_ext):
    for sep in ['_', '-', ' ']:
        fname_no_ext = fname_no_ext.replace(sep, '.')
    toks = [t for t in fname_no_ext.split('.') if t]
    for t in toks:
        if t.isdigit():
            return t
    return None

def build_student_embeddings(model_name=DF_MODEL_NAME, max_images_per_student=5):
    global student_embeddings
    student_embeddings = {}

    paths_per_id = defaultdict(list)
    for fname in os.listdir(IMAGE_DIR):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        name_no_ext = os.path.splitext(fname)[0]
        sid = parse_id_from_filename(name_no_ext)
        if sid is None:
            continue
        paths_per_id[sid].append(os.path.join(IMAGE_DIR, fname))

    students_df = load_students_df()
    id_to_name = {str(r["Id"]): str(r["Name"]) for _, r in students_df.iterrows()}

    for sid, paths in paths_per_id.items():
        paths = sorted(paths)[:max_images_per_student]
        embs = []
        for p in paths:
            try:
                reps = DeepFace.represent(img_path=p, model_name=model_name,
                                          enforce_detection=False)
                if isinstance(reps, list) and len(reps) > 0:
                    emb = np.array(reps[0]["embedding"], dtype="float32")
                elif isinstance(reps, dict) and "embedding" in reps:
                    emb = np.array(reps["embedding"], dtype="float32")
                else:
                    continue
                norm = np.linalg.norm(emb) + 1e-10
                embs.append(emb / norm)
            except Exception:
                continue
        if embs:
            centroid = np.mean(embs, axis=0)
            norm = np.linalg.norm(centroid) + 1e-10
            centroid = centroid / norm
            student_embeddings[sid] = {
                "name": id_to_name.get(sid, ""),
                "embedding": centroid
            }

def embed_face_roi_bgr(face_bgr, model_name=DF_MODEL_NAME):
    try:
        reps = DeepFace.represent(img_path=face_bgr, model_name=model_name,
                                  enforce_detection=False)
        if isinstance(reps, list) and len(reps) > 0:
            emb = np.array(reps[0]["embedding"], dtype="float32")
        elif isinstance(reps, dict) and "embedding" in reps:
            emb = np.array(reps["embedding"], dtype="float32")
        else:
            return None
        norm = np.linalg.norm(emb) + 1e-10
        return emb / norm
    except Exception:
        return None

def cosine_distance(a, b):
    return float(1.0 - float(np.dot(a, b)))

def identify_by_embedding(emb, threshold=FACENET_COSINE_THRESHOLD, margin=0.10):
    """
    emb        : normalized embedding for current face
    threshold  : max allowed distance to accept a match
    margin     : how much better the best match must be compared to 2nd best
                 (to avoid confusion between similar faces)
    """
    if emb is None or not student_embeddings:
        return None, "", float("inf")

    best_sid, best_name = None, ""
    best_dist = 1e9
    second_dist = 1e9

    # Find best and second-best distances
    for sid, entry in student_embeddings.items():
        dist = cosine_distance(emb, entry["embedding"])
        if dist < best_dist:
            second_dist = best_dist
            best_dist = dist
            best_sid = sid
            best_name = entry["name"]
        elif dist < second_dist:
            second_dist = dist

    # 1) Too far from everyone -> Unknown
    if best_dist > threshold:
        return None, "", best_dist

    # 2) Best and second best are too close -> ambiguous -> Unknown
    if second_dist - best_dist < margin:
        return None, "", best_dist

    # 3) Confident match
    return best_sid, best_name, best_dist


# ---------------- EMAIL SENDING (Central Sender) ----------------
CENTRAL_SENDER_EMAIL = ""   # your Email
CENTRAL_APP_PASSWORD = ""   # 16-char App Password

def send_attendance_email(faculty_info, attach_path):
    """
    Send attendance summary email to faculty using one central Gmail account.
    The recipient is taken from the faculty's registered email.
    """
    recipient_email = str(faculty_info.get("Email", "")).strip()
    if not recipient_email:
        return False, "No recipient email found for faculty."

    if not os.path.exists(attach_path):
        return False, f"Attachment not found: {attach_path}"

    try:
        msg = EmailMessage()
        date_s = now_date_str()
        subj = f"Attendance Summary: {date_s} — {faculty_info.get('Subject', '')}"
        msg["Subject"] = subj
        msg["From"] = CENTRAL_SENDER_EMAIL
        msg["To"] = recipient_email

        try:
            df = pd.read_csv(attach_path, dtype=str)
            count = len(df.index)
        except Exception:
            count = "Unknown"

        body = (f"Hello {faculty_info.get('Name', '')},\n\n"
                f"Here is your attendance summary for {date_s} "
                f"(Subject: {faculty_info.get('Subject', '')})\n\n"
                f"🧾 Students marked present: {count}\n\n"
                f"The detailed CSV file is attached.\n\n"
                f"Regards,\nSmart Attendance System\n")
        msg.set_content(body)

        with open(attach_path, "rb") as f:
            data = f.read()
        msg.add_attachment(data, maintype="text", subtype="csv",
                           filename=os.path.basename(attach_path))

        context = ssl.create_default_context()
        with smtplib.SMTP("smtp.gmail.com", 587, timeout=30) as server:
            server.starttls(context=context)
            server.login(CENTRAL_SENDER_EMAIL, CENTRAL_APP_PASSWORD)
            server.send_message(msg)

        return True, f"Email sent successfully to {recipient_email}."

    except smtplib.SMTPAuthenticationError:
        return False, "Authentication failed (check App Password or central Gmail)."
    except Exception as ex:
        return False, f"Error sending email: {str(ex)}"

def prompt_and_send_manual_email(faculty_info, attach_path):
    """
    Optional manual send window: lets you override the recipient email address.
    """
    dlg = tk.Toplevel(root)
    dlg.title("Send Attendance Email (Manual)")
    dlg.geometry("420x200")
    dlg.configure(bg="#f6f9fb")
    dlg.grab_set()

    tk.Label(dlg, text="Send Attendance Email Now",
             font=("Arial", 11, "bold"), bg="#f6f9fb").pack(pady=8)
    frm = tk.Frame(dlg, bg="#f6f9fb"); frm.pack(padx=10, pady=4, fill="x")

    tk.Label(frm, text="Recipient Email:", bg="#f6f9fb").grid(row=0, column=0, sticky="w", pady=6)
    email_entry = tk.Entry(frm, width=36)
    email_entry.grid(row=0, column=1, pady=6)
    email_entry.insert(0, str(faculty_info.get("Email", "")))

    status_lbl = tk.Label(dlg, text="", bg="#f6f9fb")
    status_lbl.pack(pady=6)

    def do_send():
        recipient = email_entry.get().strip()
        if not recipient:
            messagebox.showerror("Error", "Please enter recipient email.", parent=dlg)
            return
        temp_info = dict(faculty_info)
        temp_info["Email"] = recipient
        ok, msg = send_attendance_email(temp_info, attach_path)
        status_lbl.config(text=msg)
        if ok:
            messagebox.showinfo("Email", msg, parent=dlg)
            dlg.destroy()
        else:
            messagebox.showerror("Email Failed", msg, parent=dlg)

    btn_fr = tk.Frame(dlg, bg="#f6f9fb"); btn_fr.pack(pady=4)
    tk.Button(btn_fr, text="Send Now", command=do_send,
              bg="#0b3142", fg="#FFD300", width=14).grid(row=0, column=0, padx=6)
    tk.Button(btn_fr, text="Cancel", command=dlg.destroy,
              width=12).grid(row=0, column=1, padx=6)

    dlg.wait_window()

# ---------------- GUI SETUP ----------------
root = tk.Tk()
root.title("Smart Attendance System (DeepFace • SFace • SilentFace Anti-Spoof)")
root.geometry("1440x900")
root.configure(bg="#0b2846")

current_faculty = None
faculty_unlocked_until = None

def update_time_label():
    time_label.config(text=time.strftime("%d-%B-%Y %H:%M:%S"))
    root.after(1000, update_time_label)

def lock_faculty():
    global current_faculty, faculty_unlocked_until
    current_faculty = None
    faculty_unlocked_until = None
    faculty_label.config(text="Faculty: (Locked) — Unlock before taking attendance")
    load_and_populate_today_tree()

def is_unlock_valid():
    global current_faculty, faculty_unlocked_until
    if current_faculty is None or faculty_unlocked_until is None:
        return False
    if datetime.now() < faculty_unlocked_until:
        return True
    lock_faculty()
    messagebox.showwarning("Session Expired", "Unlock expired after 60 minutes. Please unlock again.")
    return False

def update_faculty_unlock_label():
    global current_faculty, faculty_unlocked_until
    if current_faculty and faculty_unlocked_until:
        now = datetime.now()
        if now >= faculty_unlocked_until:
            lock_faculty()
        else:
            remaining = faculty_unlocked_until - now
            mins, secs = divmod(int(remaining.total_seconds()), 60)
            hours, mins = divmod(mins, 60)
            faculty_label.config(
                text=f"Faculty: {current_faculty['Name']} | "
                     f"{current_faculty['Dept']} | "
                     f"{current_faculty['Subject']} "
                     f"(valid {hours:02d}:{mins:02d}:{secs:02d})"
            )
    else:
        faculty_label.config(text="Faculty: (Locked) — Unlock before taking attendance")
    root.after(1000, update_faculty_unlock_label)

# ---------------- WINDOWS ----------------
def change_password_window():
    global ADMIN_PASSWORD
    win = tk.Toplevel(root)
    win.title("Change Admin Password")
    win.geometry("380x200")
    win.configure(bg="#f8fafc")
    tk.Label(win, text="Change Admin Password",
             font=("Arial", 12, "bold"), bg="#f8fafc").pack(pady=8)

    frm = tk.Frame(win, bg="#f8fafc"); frm.pack(pady=6, padx=10, fill="x")
    tk.Label(frm, text="Current Password:", bg="#f8fafc").grid(row=0, column=0,
                                                               sticky="w", pady=6)
    cur_entry = tk.Entry(frm, show="*", width=28); cur_entry.grid(row=0, column=1, pady=6)
    tk.Label(frm, text="New Password:", bg="#f8fafc").grid(row=1, column=0,
                                                           sticky="w", pady=6)
    new_entry = tk.Entry(frm, show="*", width=28); new_entry.grid(row=1, column=1, pady=6)
    tk.Label(frm, text="Confirm New:", bg="#f8fafc").grid(row=2, column=0,
                                                          sticky="w", pady=6)
    conf_entry = tk.Entry(frm, show="*", width=28); conf_entry.grid(row=2, column=1, pady=6)

    def do_change():
        global ADMIN_PASSWORD
        cur, new, conf = cur_entry.get(), new_entry.get(), conf_entry.get()
        if not cur or not new or not conf:
            messagebox.showerror("Error", "All fields are required", parent=win); return
        if cur != ADMIN_PASSWORD:
            messagebox.showerror("Error", "Incorrect current password", parent=win); return
        if new != conf:
            messagebox.showerror("Error", "New passwords do not match", parent=win); return
        ADMIN_PASSWORD = new
        messagebox.showinfo("Success", "Admin password changed", parent=win)
        win.destroy()

    btn_fr = tk.Frame(win, bg="#f8fafc"); btn_fr.pack(pady=8)
    tk.Button(btn_fr, text="Change Password", width=16,
              bg="#0b3142", fg="#FFD300", command=do_change).grid(row=0, column=0, padx=6)
    tk.Button(btn_fr, text="Cancel", width=12,
              command=win.destroy).grid(row=0, column=1, padx=6)

def open_register_student_window():
    win = tk.Toplevel(root)
    win.title("Register New Student")
    win.geometry("420x300")
    win.configure(bg="#f1f6f9")

    tk.Label(win, text="Student Registration",
             font=("Arial", 14, "bold"), bg="#f1f6f9").pack(pady=8)
    frame = tk.Frame(win, bg="#f1f6f9"); frame.pack(pady=6)

    tk.Label(frame, text="Enter ID:", bg="#f1f6f9").grid(row=0, column=0,
                                                         sticky="w", padx=6, pady=6)
    sid = tk.Entry(frame, width=20); sid.grid(row=0, column=1, pady=6)
    tk.Label(frame, text="Enter Name:", bg="#f1f6f9").grid(row=1, column=0,
                                                           sticky="w", padx=6, pady=6)
    sname = tk.Entry(frame, width=20); sname.grid(row=1, column=1, pady=6)

    lbl_taken = tk.Label(win, text="Images Taken: 0", bg="#f1f6f9")
    lbl_taken.pack(pady=4)

    def do_take_images():
        pw = simpledialog.askstring("Admin Password", "Enter Admin Password:",
                                    show="*", parent=win)
        if pw != ADMIN_PASSWORD:
            messagebox.showerror("Access Denied", "Wrong Password!", parent=win); return
        Id = sid.get().strip(); Name = sname.get().strip()
        if not Id.isdigit() or Name == "":
            messagebox.showerror("Error", "Enter numeric ID and Name", parent=win); return

        face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
        if face_cascade.empty():
            messagebox.showerror("Error", f"Cannot load cascade: {CASCADE_PATH}", parent=win); return

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            messagebox.showerror("Error", "Cannot access camera", parent=win); return

        sampleNum = 0
        while True:
            ret, img = cap.read()
            if not ret:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            if len(faces) > 0:
                largest_w = max(w for (_, _, w, _) in faces)
                faces = [(x, y, w, h) for (x, y, w, h) in faces if w > 0.7 * largest_w]

            for (x, y, w, h) in faces:
                if FACE_MIN_SIZE <= w <= FACE_MAX_SIZE:
                    color = (0, 255, 0); msg = "Perfect distance"
                    faceROI = gray[y:y+h, x:x+w]
                    if is_face_sharp(faceROI):
                        sampleNum += 1
                        faceROI = cv2.resize(faceROI, (200, 200))
                        faceROI = cv2.equalizeHist(faceROI)
                        filename = f"{IMAGE_DIR}/{Name}.{Id}.{sampleNum}.jpg"
                        cv2.imwrite(filename, faceROI)
                        # capture feedback sound can be added here if you want
                    else:
                        msg = "Hold still (blurry)"; color = (0, 165, 255)
                else:
                    color = (0, 0, 255); msg = "Move closer/farther"

                cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
                cv2.putText(img, msg, (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.putText(img, f"Capturing {sampleNum}", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("Capturing - Press Q to stop", img)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or sampleNum >= 25:
                break

        cap.release(); cv2.destroyAllWindows()

        df = load_students_df()
        if not ((df["Id"] == Id) & (df["Name"] == Name)).any():
            df = pd.concat([df, pd.DataFrame([[Id, Name]],
                                             columns=["Id", "Name"])],
                           ignore_index=True)
            save_students_df(df)
        lbl_taken.config(text=f"Images Taken: {sampleNum}")
        total_label.config(text=f"Total Registrations till now: {len(df)}")
        messagebox.showinfo("Success", f"Saved images for {Name}", parent=win)

    tk.Button(win, text="Take Images", width=18,
              bg="#111827", fg="#FFD300", command=do_take_images).pack(pady=10)
    tk.Button(win, text="Close", width=18, command=win.destroy).pack()

def open_register_faculty_window():
    win = tk.Toplevel(root)
    win.title("Register Faculty")
    win.geometry("480x420")
    win.configure(bg="#f7fbff")

    tk.Label(win, text="Faculty Registration",
             font=("Arial", 14, "bold"), bg="#f7fbff").pack(pady=8)
    frame = tk.Frame(win, bg="#f7fbff"); frame.pack(pady=6)

    tk.Label(frame, text="Faculty ID:", bg="#f7fbff").grid(row=0, column=0,
                                                           sticky="w", padx=6, pady=6)
    fid = tk.Entry(frame, width=28); fid.grid(row=0, column=1)
    tk.Label(frame, text="Name:", bg="#f7fbff").grid(row=1, column=0,
                                                     sticky="w", padx=6, pady=6)
    fname = tk.Entry(frame, width=28); fname.grid(row=1, column=1)
    tk.Label(frame, text="Department:", bg="#f7fbff").grid(row=2, column=0,
                                                           sticky="w", padx=6, pady=6)
    fdept = tk.Entry(frame, width=28); fdept.grid(row=2, column=1)
    tk.Label(frame, text="Subject:", bg="#f7fbff").grid(row=3, column=0,
                                                        sticky="w", padx=6, pady=6)
    fsub = tk.Entry(frame, width=28); fsub.grid(row=3, column=1)
    tk.Label(frame, text="Password (for unlocking):",
             bg="#f7fbff").grid(row=4, column=0, sticky="w", padx=6, pady=6)
    fpass = tk.Entry(frame, width=28, show="*"); fpass.grid(row=4, column=1)

    tk.Label(frame, text="Email (optional):", bg="#f7fbff").grid(row=5, column=0,
                                                                 sticky="w", padx=6, pady=6)
    femail = tk.Entry(frame, width=28); femail.grid(row=5, column=1)

    def do_register_faculty():
        admin_pw = simpledialog.askstring("Admin",
                                          "Enter Admin Password to register faculty:",
                                          show="*", parent=win)
        if admin_pw is None:
            return
        if admin_pw != ADMIN_PASSWORD:
            messagebox.showerror("Error", "Incorrect Admin Password", parent=win); return
        fid_v = fid.get().strip()
        name_v = fname.get().strip()
        dept_v = fdept.get().strip()
        sub_v = fsub.get().strip()
        pass_v = fpass.get().strip()
        email_v = femail.get().strip()
        if not (fid_v and name_v and dept_v and sub_v and pass_v):
            messagebox.showerror("Error",
                                 "ID, Name, Dept, Subject and Password are required",
                                 parent=win); return
        df = load_faculty_df()
        if (df["Id"] == fid_v).any():
            messagebox.showerror("Error", "Faculty ID exists", parent=win); return
        ph = sha256_hash(pass_v)
        newrow = pd.DataFrame([[fid_v, name_v, dept_v, sub_v, ph, email_v]],
                              columns=["Id", "Name", "Dept", "Subject",
                                       "PasswordHash", "Email"])
        df = pd.concat([df, newrow], ignore_index=True)
        save_faculty_df(df)
        messagebox.showinfo("Success", f"Faculty {name_v} registered", parent=win)
        win.destroy()

    tk.Button(win, text="Register Faculty", width=20,
              bg="#0b3142", fg="#FFD300", command=do_register_faculty).pack(pady=8)
    tk.Button(win, text="Close", width=20, command=win.destroy).pack()

# ---------------- QUICK VIEW TREE ----------------
def load_and_populate_today_tree():
    for r in tree.get_children():
        tree.delete(r)
    today = now_date_str()
    df_today = load_today_attendance(today)
    if df_today.empty:
        return

    df_today["FacultyId"] = df_today["FacultyId"].astype(str)
    df_today["FacultySubject"] = df_today["FacultySubject"].astype(str)

    widths = {"Faculty": 120, "Subject": 100, "FacultyId": 80,
              "RollNo": 80, "Name": 120, "Date": 90, "Time": 80}
    cols_local = ("Faculty", "Subject", "FacultyId",
                  "RollNo", "Name", "Date", "Time")
    for c in cols_local:
        tree.heading(c, text=c)
        tree.column(c, anchor="center", width=widths[c])

    grp_cols = ["FacultyId", "FacultyName", "FacultySubject"]
    grouped = df_today.groupby(grp_cols)
    parent_counter = 0
    for (fid, fname, fsub), grp in grouped:
        date_vals = grp["Date"].dropna().unique()
        date_s = date_vals[0] if len(date_vals) > 0 else today
        try:
            times = pd.to_datetime(grp["Time"], format="%H:%M:%S", errors="coerce")
            start_time = times.min().strftime("%H:%M:%S") if times.notna().any() else ""
        except Exception:
            start_time = ""

        parent_id = f"fac_hdr_{fid}_{fsub}_{parent_counter}"
        parent_counter += 1
        tree.insert("", "end", iid=parent_id,
                    values=(fname, fsub, fid, "", "", date_s, start_time),
                    open=True, tags=("faculty_header",))

        for _, row in grp.iterrows():
            sid = str(row.get("StudentId", ""))
            sname = row.get("StudentName", "")
            date_r = row.get("Date", date_s)
            time_r = row.get("Time", "")
            tree.insert(parent_id, "end",
                        values=("", "", "", sid, sname, date_r, time_r))

    style = ttk.Style()
    style.configure("Treeview.Heading", font=("Arial", 10, "bold"))
    tree.tag_configure("faculty_header", background="#e0f0ff",
                       font=("Arial", 10, "bold"))

# ---------------- UNLOCK / ATTENDANCE ----------------
def verify_faculty_password(password):
    p_hash = sha256_hash(password)
    df = load_faculty_df()
    matched = df.loc[df["PasswordHash"] == p_hash]
    if matched.empty:
        return None
    row = matched.iloc[0]
    return {
        "Id": str(row["Id"]),
        "Name": str(row["Name"]),
        "Dept": str(row["Dept"]),
        "Subject": str(row["Subject"]),
        "Email": str(row.get("Email", "")),
        "EmailAppPassword": ""   # kept for compatibility, unused
    }

def open_unlock_dialog():
    global current_faculty, faculty_unlocked_until
    dlg = tk.Toplevel(root)
    dlg.title("Faculty Unlock")
    dlg.geometry("430x230")
    dlg.configure(bg="#f6f9fb")
    dlg.grab_set()

    tk.Label(dlg, text="Faculty Unlock (enter password and subject)",
             font=("Arial", 10, "bold"), bg="#f6f9fb").pack(pady=6)
    frm = tk.Frame(dlg, bg="#f6f9fb"); frm.pack(pady=4, padx=9, fill="x")

    tk.Label(frm, text="Password:", bg="#f6f9fb").grid(row=0, column=0,
                                                       sticky="w", pady=6)
    pw_entry = tk.Entry(frm, show="*", width=30); pw_entry.grid(row=0, column=1, pady=6)
    tk.Label(frm, text="Subject (this session):", bg="#f6f9fb").grid(row=1, column=0,
                                                                     sticky="w", pady=6)
    subj_entry = tk.Entry(frm, width=30); subj_entry.grid(row=1, column=1, pady=6)
    tk.Label(dlg, text="Tip: leave Subject empty to use registered subject.",
             font=("Arial", 8), bg="#f6f9fb").pack(pady=(0, 6))

    def do_unlock():
        global current_faculty, faculty_unlocked_until
        pw = pw_entry.get() or ""
        subj_input = subj_entry.get().strip()
        if pw == "":
            messagebox.showerror("Error", "Enter faculty password", parent=dlg); return
        faculty_info = verify_faculty_password(pw)
        if faculty_info is None:
            messagebox.showerror("Error", "Incorrect faculty password", parent=dlg); return
        chosen_subj = subj_input if subj_input != "" else faculty_info.get("Subject", "")
        faculty_info["Subject"] = chosen_subj
        current_faculty = faculty_info
        faculty_unlocked_until = datetime.now() + timedelta(minutes=60)
        dlg.destroy()
        messagebox.showinfo(
            "Unlocked",
            f"Unlocked for {current_faculty['Name']} teaching "
            f"'{current_faculty['Subject']}' until "
            f"{faculty_unlocked_until.strftime('%Y-%m-%d %H:%M:%S')}"
        )
        update_faculty_unlock_label()
        start_attendance_session()

    btn_fr = tk.Frame(dlg, bg="#f6f9fb"); btn_fr.pack(pady=6)
    tk.Button(btn_fr, text="Unlock & Start", width=14,
              bg="#0b3142", fg="#FFD300", command=do_unlock).grid(row=0, column=0, padx=6)
    tk.Button(btn_fr, text="Cancel", width=10,
              command=dlg.destroy).grid(row=0, column=1, padx=6)
    pw_entry.focus_set()
    dlg.wait_window()

def take_attendance_button_handler():
    global current_faculty, faculty_unlocked_until
    if current_faculty and faculty_unlocked_until and datetime.now() < faculty_unlocked_until:
        start_attendance_session()
    else:
        open_unlock_dialog()

def start_attendance_session():
    global current_faculty, faculty_unlocked_until
    if current_faculty is None or faculty_unlocked_until is None or datetime.now() >= faculty_unlocked_until:
        messagebox.showerror("Locked", "Attendance locked or unlock expired.")
        return
    if not MP_AVAILABLE:
        messagebox.showerror("Error",
                             "Mediapipe not installed. "
                             "Install using 'pip install mediapipe' and retry.")
        return
    if silentface_liveness.model is None or not silentface_liveness.model_files:
        messagebox.showerror(
            "Anti-Spoof Error",
            "SilentFace models not loaded.\n"
            "Check folder resources/anti_spoof_models and imports."
        )
        return

    build_student_embeddings(model_name=DF_MODEL_NAME)
    if not student_embeddings:
        messagebox.showerror("Error", "No student embeddings built. Register students first.")
        return

    today = now_date_str()
    df_today = load_today_attendance(today)
    if not df_today.empty:
        marked_keys = set((str(r["FacultyId"]),
                           str(r["FacultySubject"]),
                           str(r["StudentId"]))
                          for _, r in df_today.iterrows())
    else:
        marked_keys = set()

    load_and_populate_today_tree()

    session_faculty = dict(current_faculty)

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=4,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        messagebox.showerror("Error", "Cannot access camera")
        return

    tracks = {}
    next_tid = 0
    frame_idx = 0

    window_name = "Taking Attendance (DeepFace • SFace • SilentFace Anti-Spoof) - Press Q to Quit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)

    try:
        while True:
            if faculty_unlocked_until and datetime.now() >= faculty_unlocked_until:
                messagebox.showinfo("Session Ended",
                                    "Faculty unlock expired (1 hour). Session will stop.")
                break

            ret, img = cap.read()
            if not ret:
                continue
            frame_idx += 1
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            detections = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

            if len(detections) > 0:
                largest_w = max(w for (_, _, w, _) in detections)
                detections = [(x, y, w, h) for (x, y, w, h) in detections
                              if w > 0.7 * largest_w]

            det_list = []
            for (x, y, w, h) in detections:
                cx, cy = x + w//2, y + h//2
                det_list.append((x, y, w, h, (cx, cy)))

            for det in det_list:
                x, y, w, h, centroid = det

                # Distance gate
                if w < FACE_MIN_SIZE or w > FACE_MAX_SIZE:
                    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    cv2.putText(img, "Adjust your distance", (x, y-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                    normalized = (w - FACE_MIN_SIZE) / max(1.0, (FACE_MAX_SIZE - FACE_MIN_SIZE))
                    normalized = max(0.0, min(1.0, normalized))
                    bar_x1, bar_y1 = x, y+h+20
                    bar_y2 = bar_y1 + 10
                    bar_x2 = x + int(normalized * w)
                    cv2.rectangle(img, (bar_x1, bar_y1),
                                  (bar_x1+w, bar_y2), (50, 50, 50), 2)
                    cv2.rectangle(img, (bar_x1, bar_y1),
                                  (bar_x2, bar_y2), (0, 0, 255), -1)
                    cv2.putText(img, "Too close/far", (x, y+h+45),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    continue

                # Track assignment
                best_tid, best_dist = None, None
                for tid, t in tracks.items():
                    d = euclidean(centroid, t['centroid'])
                    if d <= MATCH_DISTANCE_PX and (best_tid is None or d < best_dist):
                        best_tid, best_dist = tid, d
                if best_tid is None:
                    tid = next_tid
                    next_tid += 1
                    tracks[tid] = {
                        'centroid': centroid,
                        'last_seen': frame_idx,
                        'preds': deque(maxlen=FRAME_THRESHOLD),
                        'recognized': False,
                        'pending_sid': None,
                        'blink_counter': 0,
                        'blinked': False,
                        'marked': False,
                        'last_beep': 0.0,

                        # liveness-related flags
                        'live_peak_reached': False,    # once score hits ~0.99 (star)
                        'spoof_flag': False,           # permanent spoof if <= 0.23 in good light
                        'last_live_score': 0.0,        # last SilentFace score
                        'last_label_live': 0, 
                         # 🔊 NEW: one-time TTS flags
                        'spoof_announced': False,
                        'unreg_announced': False,
                        # last SilentFace label
                    }
                    best_tid = tid
                t = tracks[best_tid]
                t['centroid'] = centroid
                t['last_seen'] = frame_idx

                # ---------- Brightness & Sharpness check ----------
                roi_gray = gray[y:y+h, x:x+w]

                # Mean brightness of face region (0–255)
                brightness = float(np.mean(roi_gray))

                if brightness < BRIGHTNESS_MIN:
                    # Too dark: ask user to increase light, but DO NOT mark spoof
                    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 255), 2)
                    cv2.putText(
                        img,
                        "Too dark - increase lighting",
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2,
                    )

                    normalized = (w - FACE_MIN_SIZE) / max(1.0, (FACE_MAX_SIZE - FACE_MIN_SIZE))
                    normalized = max(0.0, min(1.0, normalized))
                    bar_x1, bar_y1 = x, y+h+20
                    bar_y2 = bar_y1 + 10
                    bar_x2 = x + int(normalized * w)
                    cv2.rectangle(img, (bar_x1, bar_y1),
                                  (bar_x1 + w, bar_y2), (50, 50, 50), 2)
                    cv2.rectangle(img, (bar_x1, bar_y1),
                                  (bar_x2, bar_y2), (0, 255, 255), -1)
                    cv2.putText(img, "Add more light", (x, y+h+45),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                    # Skip liveness/DeepFace/blink when too dark
                    continue

                # If brightness is OK, then check blur
                if not is_face_sharp(roi_gray):
                    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 165, 255), 2)
                    cv2.putText(img, "Hold still (blurry)", (x, y-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

                    normalized = (w - FACE_MIN_SIZE) / max(1.0, (FACE_MAX_SIZE - FACE_MIN_SIZE))
                    normalized = max(0.0, min(1.0, normalized))
                    bar_x1, bar_y1 = x, y+h+20
                    bar_y2 = bar_y1 + 10
                    bar_x2 = x + int(normalized * w)
                    cv2.rectangle(img, (bar_x1, bar_y1),
                                  (bar_x1+w, bar_y2), (50, 50, 50), 2)
                    cv2.rectangle(img, (bar_x1, bar_y1),
                                  (bar_x2, bar_y2), (0, 165, 255), -1)
                    cv2.putText(img, "Adjust slightly", (x, y+h+45),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
                    continue

                # ---------- REAL ANTI-SPOOF USING SILENTFACE ----------
                label_live, live_score = silentface_liveness.score(img, [x, y, w, h])
                t['last_live_score'] = float(live_score)
                t['last_label_live'] = int(label_live)

                # 1) If score <= 0.23 -> permanent spoof ONLY if brightness was OK
                if live_score <= 0.23:
                    # Here brightness is already >= BRIGHTNESS_MIN because we continued above for dark
                    if not t.get('spoof_flag', False):
                        t['spoof_flag'] = True
                         # 🔊 Speak "spoof" only the first time this track is marked spoof
                        if not t.get('spoof_announced', False):
                            speak_name_async("spoof")
                            t['spoof_announced'] = True    
                # If permanently spoofed, block everything for this track
                if t.get('spoof_flag', False):
                    live_text = f"SPOOF {live_score:.2f} (blocked)"
                    live_color = (0, 0, 255)
                    cv2.putText(
                        img,
                        live_text,
                        (x, y + h + 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        live_color,
                        2
                    )
                    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    # Do not run DeepFace / blink / attendance
                    continue

                # 2) Display logic + star (peak = 1.0) logic
                if live_score >= SILENTFACE_THRESHOLD and label_live == 1:
                    # Face passes LIVE threshold for this frame
                    # ⭐ One-time “star condition”: if score ever reaches ~1.0
                    if live_score >= 0.99:
                        t['live_peak_reached'] = True

                    live_text = f"LIVE {live_score:.2f}"
                    live_color = (0, 255, 0)
                elif live_score > 0.0:
                    # 0 < score < 0.7 => SPOOF (non-permanent)
                    live_text = f"SPOOF {live_score:.2f}"
                    live_color = (0, 0, 255)
                else:
                    # live_score == 0 already handled above, but keep label
                    live_text = f"SPOOF {live_score:.2f}"
                    live_color = (0, 0, 255)

                cv2.putText(
                    img,
                    live_text,
                    (x, y + h + 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    live_color,
                    2
                )

                # If current frame is not LIVE, skip recognition / blink / attendance
                if not (live_score >= SILENTFACE_THRESHOLD and label_live == 1):
                    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    continue

                # ---------- If LIVE: DeepFace embedding + identify ----------
                roi_bgr = img[y:y+h, x:x+w]
                emb = embed_face_roi_bgr(roi_bgr, DF_MODEL_NAME)
                sid_pred, name_pred, dist = identify_by_embedding(emb, FACENET_COSINE_THRESHOLD)

                # queue predictions for stability
                t['preds'].append(sid_pred if sid_pred is not None else None)

                # accept after FRAME_THRESHOLD consistent frames
                if not t['recognized'] and len(t['preds']) == FRAME_THRESHOLD:
                    preds_list = list(t['preds'])
                    if None not in preds_list:
                        unanimous = all(p == preds_list[0] for p in preds_list)
                        if unanimous:
                            t['recognized'] = True
                            t['pending_sid'] = preds_list[0]

                # Draw box + label (with star + blink)
                box_color = (0, 165, 255)
                label = "Recognizing..."
                sid_show = ""

                if t['recognized']:
                    sid_show = t['pending_sid']
                    if sid_show is not None:
                        label = student_embeddings.get(sid_show, {}).get("name", "Student")

                        # ⭐ Show star if peak live score has been seen once
                        if t.get('live_peak_reached', False):
                            label += " *"

                        # 👁 Show blink info
                        if t.get('blinked', False):
                            label += " (blinked)"

                        box_color = (0, 255, 0)
                else:
                    if sid_pred is None:
                        box_color = (0, 0, 255)
                        label = "Unregister"
                        # 🔊 Speak "unregister" only once per track
                        if not t.get('unreg_announced', False):
                           speak_name_async("unregister")
                           t['unreg_announced'] = True
                cv2.rectangle(img, (x, y), (x+w, y+h), box_color, 2)
                cv2.putText(img, f"{label} [{sid_show}]", (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)

                # Distance meter (visual only)
                normalized = (w - FACE_MIN_SIZE) / max(1.0, (FACE_MAX_SIZE - FACE_MIN_SIZE))
                normalized = max(0.0, min(1.0, normalized))
                mid = (FACE_MIN_SIZE + FACE_MAX_SIZE) / 2.0
                if w < FACE_MIN_SIZE*0.9 or w > FACE_MAX_SIZE*1.1:
                    bar_color = (0, 0, 255)
                    msg = "Too close/far"
                elif abs(w - mid) > 40:
                    bar_color = (0, 165, 255)
                    msg = "Adjust slightly"
                else:
                    bar_color = (0, 255, 0)
                    msg = "Perfect distance"
                    # we removed beep here to avoid default laptop sound

                bar_x1 = x
                bar_y1 = y+h+20
                bar_y2 = bar_y1 + 10
                bar_x2 = x + int(normalized * w)
                cv2.rectangle(img, (bar_x1, bar_y1),
                              (bar_x1+w, bar_y2), (50, 50, 50), 2)
                cv2.rectangle(img, (bar_x1, bar_y1),
                              (bar_x2, bar_y2), bar_color, -1)
                cv2.putText(img, msg, (x, y+h+45),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, bar_color, 2)

            # ---- Blink detection (FaceMesh) ----
            img_h, img_w = img.shape[:2]
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            landmarks_entries = []
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    lm_coords = []
                    xs = []; ys = []
                    for lm in face_landmarks.landmark:
                        x_px = int(lm.x * img_w)
                        y_px = int(lm.y * img_h)
                        lm_coords.append((x_px, y_px))
                        xs.append(x_px)
                        ys.append(y_px)
                    centroid = (int(np.mean(xs)), int(np.mean(ys)))
                    landmarks_entries.append({'lm_coords': lm_coords,
                                              'centroid': centroid})

            for tid, t in list(tracks.items()):
                if 'centroid' not in t:
                    continue
                # nearest mesh to this track
                assigned, md = None, None
                for entry in landmarks_entries:
                    d = euclidean(t['centroid'], entry['centroid'])
                    if md is None or d < md:
                        md = d
                        assigned = entry
                if assigned is None:
                    continue

                lm_coords = assigned['lm_coords']
                left_ear = compute_ear_safe(lm_coords, LEFT_EYE_IDX)
                right_ear = compute_ear_safe(lm_coords, RIGHT_EYE_IDX)
                avg_ear = (left_ear + right_ear) / 2.0

                if avg_ear < EAR_THRESHOLD:
                    t['blink_counter'] += 1
                else:
                    if t['blink_counter'] >= EAR_CONSEC_FRAMES:
                        t['blinked'] = True
                    t['blink_counter'] = 0

                # If recognized + blinked + proper liveness -> mark attendance
                if (
                    t.get('pending_sid') and
                    t.get('blinked') and
                    not t.get('marked', False) and
                    not t.get('spoof_flag', False) and
                    t.get('live_peak_reached', False) and
                    (t.get('last_live_score', 0.0) >= SILENTFACE_THRESHOLD) and
                    (t.get('last_label_live', 0) == 1)
                ):
                    sid = t['pending_sid']
                    key = (
                        str(current_faculty["Id"]),
                        str(current_faculty["Subject"]),
                        str(sid)
                    )
                    if key not in marked_keys:
                        sname = student_embeddings.get(sid, {}).get("name", "")
                        row = [
                            current_faculty["Id"],
                            current_faculty["Name"],
                            current_faculty["Dept"],
                            current_faculty["Subject"],
                            sid,
                            sname,
                            today,
                            now_time_str()
                        ]
                        append_to_today_attendance(row)
                        load_and_populate_today_tree()
                        marked_keys.add(key)
                        t['marked'] = True

                        # Speak student's name instead of beep
                        speak_name_async(f"{sname} present")

                        messagebox.showinfo(
                            "Attendance",
                            f"Marked: {sname} [{sid}] at {row[-1]}"
                        )

            # prune old tracks
            to_del = [tid for tid, t in tracks.items()
                      if frame_idx - t['last_seen'] > TRACK_INACTIVE_FRAMES]
            for tid in to_del:
                try:
                    del tracks[tid]
                except Exception:
                    pass

            cv2.imshow(window_name, img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        try:
            face_mesh.close()
        except Exception:
            pass
        cap.release()
        cv2.destroyAllWindows()

        # AFTER attendance session ends → try auto-send email
        try:
            attach_path = attendance_file_path(today)
            faculty_info_local = dict(session_faculty)  # snapshot

            if faculty_info_local.get("Email", "").strip():
                ok, msg = send_attendance_email(faculty_info_local, attach_path)
                if ok:
                    messagebox.showinfo(
                        "Email Sent",
                        f"Attendance email sent to {faculty_info_local.get('Email')}"
                    )
                else:
                    res = messagebox.askyesno(
                        "Email NOT Sent",
                        f"Auto-send failed: {msg}\n\n"
                        f"Do you want to enter email manually and resend?"
                    )
                    if res:
                        prompt_and_send_manual_email(faculty_info_local, attach_path)
            else:
                res = messagebox.askyesno(
                    "No Email Found",
                    "No email saved for this faculty.\n"
                    "Do you want to enter it now to send the attendance?"
                )
                if res:
                    prompt_and_send_manual_email(faculty_info_local, attach_path)

        except Exception as e:
            messagebox.showerror("Email Error",
                                 f"Unexpected error while attempting to send email: {e}")

    messagebox.showinfo("Attendance",
                        "Attendance session finished and saved for today.")

# ---------------- VIEW ATTENDANCE ----------------
def open_view_attendance_window():
    win = tk.Toplevel(root)
    win.title("View Today's Attendance")
    win.geometry("900x520")
    win.configure(bg="#ffffff")
    tk.Label(win, text="Today's Attendance (Grouped by Faculty / Subject)",
             font=("Arial", 12, "bold"), bg="#ffffff").pack(pady=6)

    cols = ("Faculty", "Subject", "FacultyId", "RollNo",
            "Name", "Date", "Time")
    treev = ttk.Treeview(win, columns=cols, show="headings", height=18)
    widths = {"Faculty": 120, "Subject": 100, "FacultyId": 80,
              "RollNo": 80, "Name": 130, "Date": 90, "Time": 80}
    for c in cols:
        treev.heading(c, text=c)
        treev.column(c, anchor="center", width=widths[c])
    treev.pack(fill="both", expand=True, padx=10, pady=6)

    today = now_date_str()
    df = load_today_attendance(today)
    if df.empty:
        tk.Label(win, text="No attendance recorded for today.",
                 bg="#ffffff").pack(pady=10)
        tk.Button(win, text="Close", command=win.destroy,
                  width=18).pack(pady=6)
        return

    df["FacultyId"] = df["FacultyId"].astype(str)
    df["FacultySubject"] = df["FacultySubject"].astype(str)
    grouped = df.groupby(["FacultyId", "FacultyName", "FacultySubject"])

    parent_counter = 0
    for (fid, fname, fsub), grp in grouped:
        date_vals = grp["Date"].dropna().unique()
        date_s = date_vals[0] if len(date_vals) > 0 else today
        try:
            times = pd.to_datetime(grp["Time"], format="%H:%M:%S", errors="coerce")
            start_time = times.min().strftime("%H:%M:%S") if times.notna().any() else ""
        except Exception:
            start_time = ""

        hdr_id = f"hdr_{fid}_{fsub}_{parent_counter}"
        parent_counter += 1
        treev.insert("", "end", iid=hdr_id,
                     values=(fname, fsub, fid, "", "", date_s, start_time),
                     tags=("faculty_header",))

        for _, row in grp.iterrows():
            sid = str(row.get("StudentId", ""))
            sname = row.get("StudentName", "")
            date_r = row.get("Date", date_s)
            time_r = row.get("Time", "")
            treev.insert(hdr_id, "end",
                         values=("", "", "", sid, sname, date_r, time_r))

    style = ttk.Style()
    style.configure("Treeview.Heading", font=("Arial", 10, "bold"))
    treev.tag_configure("faculty_header", background="#e0f0ff",
                        font=("Arial", 10, "bold"))
    tk.Button(win, text="Close", command=win.destroy,
              width=18).pack(pady=6)

# ---------------- DASHBOARD ----------------
banner = tk.Frame(root, bg="#072540", height=70)
banner.pack(fill="x")
tk.Label(
    banner,
    text=" Attendance Management System (DeepFace • SFace • SilentFace Anti-Spoof) ",
    font=("Helvetica", 18, "bold"),
    bg="#072540",
    fg="#ffd300"
).pack(pady=14)

time_label = tk.Label(root, text="", font=("Arial", 11),
                      bg="#0b2846", fg="#ffffff")
time_label.pack()
update_time_label()

faculty_label = tk.Label(
    root,
    text="Faculty: (Locked) — Unlock before taking attendance",
    font=("Arial", 12, "bold"),
    bg="#0b2846",
    fg="#ffffff"
)
faculty_label.pack(pady=6)
update_faculty_unlock_label()

buttons_frame = tk.Frame(root, bg="#0b2846")
buttons_frame.pack(pady=18)

btn_style = {
    "font": ("Arial", 12, "bold"),
    "width": 19,
    "height": 3,
    "bg": "#111827",
    "fg": "#FFD300",
    "relief": "raised",
    "bd": 3,
}

tk.Button(
    buttons_frame,
    text="Register New Student",
    command=open_register_student_window,
    **btn_style
).grid(row=0, column=0, padx=12, pady=12)

tk.Button(
    buttons_frame,
    text="Register Faculty",
    command=open_register_faculty_window,
    **btn_style
).grid(row=0, column=1, padx=12, pady=12)

tk.Button(
    buttons_frame,
    text="Take Attendance",
    command=take_attendance_button_handler,
    **btn_style
).grid(row=0, column=2, padx=12, pady=12)

tk.Button(
    buttons_frame,
    text="View Attendance",
    command=open_view_attendance_window,
    **btn_style
).grid(row=1, column=0, padx=12, pady=12)

# ---------------- SEND FULL ATTENDANCE EMAIL ----------------
def open_send_attendance_to_anyone():
    """
    Allows admin to send today's complete attendance CSV file
    (including all faculty & students) to any recipient email.
    Uses the central Gmail credentials.
    """
    win = tk.Toplevel(root)
    win.title("Send Today's Attendance CSV")
    win.geometry("420x200")
    win.configure(bg="#f5f9fc")
    win.grab_set()

    tk.Label(win, text="Send Full Attendance Report",
             font=("Arial", 12, "bold"), bg="#f5f9fc").pack(pady=10)
    frm = tk.Frame(win, bg="#f5f9fc"); frm.pack(pady=6)

    tk.Label(frm, text="Recipient Email:", bg="#f5f9fc").grid(row=0, column=0,
                                                              sticky="w", padx=5)
    recipient_entry = tk.Entry(frm, width=32, font=("Arial", 11))
    recipient_entry.grid(row=0, column=1, pady=6)

    def do_send():
        recipient = recipient_entry.get().strip()
        if not recipient:
            messagebox.showerror("Missing Email", "Please enter recipient email.",
                                 parent=win)
            return

        today = now_date_str()
        attach_path = attendance_file_path(today)
        if not os.path.exists(attach_path):
            messagebox.showerror("No Attendance File",
                                 f"No attendance CSV found for today ({today}).",
                                 parent=win)
            return

        try:
            msg = EmailMessage()
            msg["Subject"] = f"Full Attendance Report — {today}"
            msg["From"] = CENTRAL_SENDER_EMAIL
            msg["To"] = recipient

            try:
                df = pd.read_csv(attach_path, dtype=str)
                total_rows = len(df.index)
            except Exception:
                total_rows = "Unknown"

            msg.set_content(
                f"Hello,\n\nAttached is the complete attendance report for {today}.\n"
                f"Total Records: {total_rows}\n\nRegards,\nSmart Attendance System"
            )

            with open(attach_path, "rb") as f:
                data = f.read()
            msg.add_attachment(data, maintype="text", subtype="csv",
                               filename=os.path.basename(attach_path))

            context = ssl.create_default_context()
            with smtplib.SMTP("smtp.gmail.com", 587, timeout=30) as server:
                server.starttls(context=context)
                server.login(CENTRAL_SENDER_EMAIL, CENTRAL_APP_PASSWORD)
                server.send_message(msg)

            messagebox.showinfo("Success",
                                f"Attendance CSV sent successfully to:\n{recipient}",
                                parent=win)
            win.destroy()

        except smtplib.SMTPAuthenticationError:
            messagebox.showerror("Authentication Failed",
                                 "Invalid App Password or Gmail login failed.",
                                 parent=win)
        except Exception as e:
            messagebox.showerror("Error",
                                 f"Failed to send email:\n{e}", parent=win)

    tk.Button(win, text="Send", bg="#0b3142", fg="#FFD300",
              width=10, command=do_send).pack(pady=10)
    tk.Button(win, text="Cancel", command=win.destroy,
              width=10).pack()

tk.Button(
    buttons_frame,
    text="Send Attendance Email",
    command=open_send_attendance_to_anyone,
    **btn_style
).grid(row=1, column=1, padx=12, pady=12)

tk.Button(
    buttons_frame,
    text="Change Admin Password",
    command=change_password_window,
    **btn_style
).grid(row=1, column=2, padx=12, pady=12)

left_panel = tk.LabelFrame(root, text="Today's Attendance (Quick view)",
                           font=("Arial", 14, "bold"),
                           bg="#0b2846", fg="#fada3a")
left_panel.place(x=410, y=390, width=720, height=300)
cols = ("Faculty", "Subject", "FacultyId", "RollNo", "Name", "Date", "Time")
tree = ttk.Treeview(left_panel, columns=cols, show="headings", height=10)
widths = {"Faculty": 120, "Subject": 100, "FacultyId": 80,
          "RollNo": 80, "Name": 120, "Date": 90, "Time": 80}
for c in cols:
    tree.heading(c, text=c)
    tree.column(c, anchor="center", width=widths[c])
tree.pack(fill="both", expand=True, pady=6, padx=6)

tk.Button(root, text="Lock Now", bg="#ff7f50",
          command=lock_faculty).place(x=650, y=760)
tk.Button(root, text="Quit", bg="#d90429", fg="#fff",
          command=root.quit).place(x=750, y=760)

students_df = load_students_df()
total_label = tk.Label(root,
                       text=f"Total Registrations till now: {len(students_df)}",
                       bg="#0b2846", fg="#fff", font=("Arial", 12, "bold"))
total_label.place(x=600, y=350)

load_and_populate_today_tree()

root.mainloop()
