# main.py
import cv2
import numpy as np

def load_video(file_path):
    cap = cv2.VideoCapture(file_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def preprocess_frame(frame):
    return cv2.GaussianBlur(frame, (5, 5), 0)

def remove_background(frames):
    subtractor = cv2.createBackgroundSubtractorMOG2()
    masks = []
    for frame in frames:
        fg_mask = subtractor.apply(frame)
        masks.append(fg_mask)
    return masks

def compute_optical_flow(frames):
    flow_visualizations = []
    hsv_mask = np.zeros_like(frames[0]) # Sets saturation to max 255
    hsv_mask[..., 1] = 255
    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)

    for i in range(1, len(frames)):
        gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 
                                            0.5, 3, 15, 3, 5, 1.2, 0) # Turns every frame into grayscale
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv_mask[..., 0] = ang * 180 / np.pi / 2
        hsv_mask[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        rgb_representation = cv2.cvtColor(hsv_mask, cv2.COLOR_HSV2BGR) # Direction (hue) and speed (brightness)
        flow_visualizations.append(rgb_representation)
        prev_gray = gray
    return flow_visualizations
