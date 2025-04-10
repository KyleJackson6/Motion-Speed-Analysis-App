# app.py
import streamlit as st
import tempfile
import cv2
import numpy as np
from main import load_video, preprocess_frame, remove_background, compute_optical_flow

def display_frames(title, frames, max_frames=5, with_slider=False):
    st.subheader(title)
    if with_slider and len(frames) > 0:
        frame_idx = st.slider(f"Select a frame from {title}", 0, len(frames)-1, 0)
        st.image(frames[frame_idx], channels="BGR")
    else:
        for i in range(0, len(frames), max(len(frames) // max_frames, 1)):
            st.image(frames[i], channels="BGR")

def show_speed_graph(optical_flows):
    st.subheader("Average Motion Speed per Frame")
    speeds = []
    for flow_img in optical_flows:
        gray = cv2.cvtColor(flow_img, cv2.COLOR_BGR2GRAY)
        avg_speed = np.mean(gray)
        speeds.append(avg_speed)
    st.line_chart(speeds)

def main():
    st.title("Motion Speed Analyzer")
    uploaded_file = st.file_uploader("Upload a video file (.mp4)", type=["mp4"])

    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        
        with st.spinner('Loading and processing video...'):
            frames = load_video(tfile.name)
            sampled_frames = frames[::5]  # Sample every 5th frame

            preprocessed = [preprocess_frame(f) for f in sampled_frames]
            masks = remove_background(preprocessed)
            optical_flows = compute_optical_flow(preprocessed)

        display_frames("Original Frames", sampled_frames, with_slider=True)
        display_frames("Motion Masks", [cv2.cvtColor(m, cv2.COLOR_GRAY2BGR) for m in masks], with_slider=True)
        display_frames("Optical Flow Visualization", optical_flows, with_slider=True)

        show_speed_graph(optical_flows)

        st.download_button(
            label="Download Sample Video",
            data=open(tfile.name, "rb").read(),
            file_name="processed_video.mp4",
            mime="video/mp4"
        )

if __name__ == '__main__':
    main()