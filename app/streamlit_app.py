import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import cv2
from collections import deque

from src.detection.face_detector import detect_face
from src.detection.emotion_predictor import predict_emotion
from src.recommendation.music_recommender import recommend_music

# ================= UI =================
emoji_map = {
    "happy": "😄",
    "sad": "😢",
    "angry": "😠",
    "neutral": "😐",
    "surprise": "😲",
    "fear": "😨",
    "disgust": "🤢"
}

st.markdown(
    "<h1 style='text-align: center; color: #00FFAA;'>🎭 Emotion Music AI</h1>",
    unsafe_allow_html=True
)

st.sidebar.title("Controls")
run = st.sidebar.checkbox("Start Camera")

genre = st.sidebar.selectbox(
    "Select Mood Type",
    ["Chill", "Party", "Focus"]
)

refresh = st.sidebar.button("🔄 Refresh Songs")

FRAME_WINDOW = st.image([])
emotion_placeholder = st.empty()
song_placeholder = st.empty()

# ================= CAMERA =================
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

frame_count = 0
emotion_buffer = deque(maxlen=3)
confidence_buffer = deque(maxlen=3)

# ================= MAIN LOOP =================
while run:
    ret, frame = camera.read()
    if not ret:
        break

    # 🔥 Adaptive brightness (FIXED)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if gray.mean() < 80:
        frame = cv2.convertScaleAbs(frame, alpha=1.1, beta=10)

    frame = cv2.resize(frame, (480, 360))
    frame_count += 1

    if frame is None or frame.size == 0:
        continue

    face = detect_face(frame)

    # ================= PREDICTION =================
    if face is not None and frame_count % 5 == 0:
        face = cv2.resize(face, (128, 128))   # match training size
        emotion, confidence = predict_emotion(face)

        emotion_buffer.append(emotion)
        confidence_buffer.append(confidence)

    # ================= OUTPUT =================
    if len(emotion_buffer) > 0 and len(confidence_buffer) > 0:

        final_emotion = emotion_buffer[-1]
        songs = recommend_music(final_emotion)

        avg_conf = sum(confidence_buffer) / len(confidence_buffer)

        if frame_count % 5 == 0:

        # 🔥 GROUP EVERYTHING FOR EMOTION HERE
            with emotion_placeholder.container():
                st.subheader(
                    f"{emoji_map[final_emotion]} {final_emotion.upper()}"
                )

                if avg_conf > 0.4:
                    st.progress(float(avg_conf))
                    st.caption(f"Confidence: {avg_conf*100:.2f}%")

        # 🔥 SONGS (SEPARATE PLACEHOLDER)
            with song_placeholder.container():
                st.write("🎵 Recommended Songs:")
                for song in songs[:5]:
                    st.image(song['image'], width=150)
                    
                    st.markdown(
                        f"🎧 **{song['name']} - {song['artist']}**  \n"
                        f"[▶️ Play on Spotify]({song['url']})"
                    )

    FRAME_WINDOW.image(frame, channels="BGR")

camera.release()