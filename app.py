import os
import cv2
import av
import pandas as pd
from datetime import datetime
import streamlit as st
from deepface import DeepFace
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import uuid

# Paths
DB_PATH = "faces_db"
ATTENDANCE_FILE = "attendance.csv"
os.makedirs(DB_PATH, exist_ok=True)

# Ensure attendance file exists
if not os.path.exists(ATTENDANCE_FILE):
    pd.DataFrame(columns=["Name", "Time"]).to_csv(ATTENDANCE_FILE, index=False)


# Attendance marker function
def mark_attendance(name):
    df = pd.read_csv(ATTENDANCE_FILE)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Avoid duplicate marking in same day
    if not ((df["Name"] == name) & (df["Time"].str.contains(datetime.now().strftime("%Y-%m-%d")))).any():
        df.loc[len(df)] = [name, now]
        df.to_csv(ATTENDANCE_FILE, index=False)


# Video transformer for streamlit-webrtc
class FaceRecognitionTransformer(VideoTransformerBase):
    def __init__(self):
        self.last_name = None

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Save temp frame for DeepFace
        import uuid
        temp_img = f"temp_{uuid.uuid4().hex}.jpg"
        cv2.imwrite(temp_img, img)
        try:
            result = DeepFace.find(img_path=temp_img, db_path=DB_PATH, enforce_detection=False, silent=True)
            if result and not result[0].empty:
                identity = os.path.basename(result[0].iloc[0]["identity"])
                name = os.path.splitext(identity)[0]
                self.last_name = name

                mark_attendance(name)

                cv2.putText(img, f"{name}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                cv2.putText(img, "Unknown", (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 0, 255), 2, cv2.LINE_AA)

        except Exception:
            cv2.putText(img, "No face detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2, cv2.LINE_AA)

        # Clean up temp image file
        if os.path.exists(temp_img):
            os.remove(temp_img)

        return img

# Streamlit UI
st.title("📸 Real-Time Attendance System (DeepFace + WebRTC)")

menu = st.sidebar.radio("Menu", ["Register", "Mark Attendance (Live)", "View Attendance"])

# Register new user
if menu == "Register":
    st.subheader("Register a New User")
    name = st.text_input("Enter your name")
    uploaded_file = st.file_uploader("Upload your photo", type=["jpg", "jpeg", "png"])

    if uploaded_file and name:
        save_path = os.path.join(DB_PATH, f"{name}.jpg")
        with open(save_path, "wb") as f:
            f.write(uploaded_file.read())
        st.success(f"{name} registered successfully ✅")

# Mark attendance with live webcam
elif menu == "Mark Attendance (Live)":
    st.subheader("Mark Attendance via Webcam (Real-Time)")
    webrtc_streamer(key="attendance", video_transformer_factory=FaceRecognitionTransformer)

# View attendance records
elif menu == "View Attendance":
    st.subheader("Attendance Records")
    df = pd.read_csv(ATTENDANCE_FILE)
    st.dataframe(df)
