import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import cv2
import time
from audio_recorder_streamlit import audio_recorder

st.set_page_config(layout="wide")

# st.title("Hello World!")

col1, col2 = st.columns(2)

with col1:
    lecture_placeholder = st.empty()

with col2:
    placeholder = st.empty()
    audio_bytes = audio_recorder(text="")
    if audio_bytes:
        st.audio(audio_bytes, format="audio/wav")

capture = cv2.VideoCapture(0)
lecture_capture = cv2.VideoCapture("lecture.mp4")

while True:
    ret, frame = capture.read()
    ret, video_frame = lecture_capture.read()
    placeholder.image(frame, caption="")
    lecture_placeholder.image(video_frame, caption="")
