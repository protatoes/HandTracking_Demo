import time
import threading

import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
from matplotlib import pyplot as plt

import mediapipe as mp
from PIL import Image
import cv2
from HandModule import handDetector

#Set some header text
st.title("Digital Biomarkers Orthogonal Validaiton Demo")
st.write("Some text here")


lock = threading.Lock()
img_container = {"img": None}


#sample video call back that 
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    with lock:
        img_container["img"] = img

    return frame


ctx = webrtc_streamer(key="example",video_frame_callback=video_frame_callback)

fig_place = st.empty()
fig, ax = plt.subplots(1, 1)

while ctx.state.playing:
    with lock:
        img = img_container["img"]
    if img is None:
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ax.cla()
    ax.hist(gray.ravel(), 256, [0, 256])
    fig_place.pyplot(fig)