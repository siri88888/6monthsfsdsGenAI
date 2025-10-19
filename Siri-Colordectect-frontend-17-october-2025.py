import streamlit as st
import cv2
import numpy as np

st.set_page_config(page_title="Vertical Color Detection", layout="wide")

# --- Background Styling ---
DEFAULT_BG = "https://64.media.tumblr.com/911052856c0d258c45ef61c36379ae29/2dade201121f6c1f-88/s1280x1920/5bce19b73bd62da18c2e139637d7b3cf6eee093d.jpg"

st.markdown(f"""
<style>
[data-testid="stAppViewContainer"] {{
  background-image: linear-gradient(135deg, #ff8a00 0%, #ffd59a 100%), url('{DEFAULT_BG}');
  background-size: cover;
  background-position: center;
  background-repeat: no-repeat;
  background-blend-mode: overlay;
}}
[data-testid="stHeader"] {{
  background: rgba(0,0,0,0);
}}
.stButton>button {{
  width: 100%;
  padding: 12px;
  margin-top: 10px;
  border: none;
  border-radius: 12px;
  font-weight: bold;
  font-size: 16px;
  box-shadow: 0 4px 10px rgba(0,0,0,0.2);
  color: white;
}}
#red-btn>button {{
  background: linear-gradient(90deg, #ff0000, #ff6666);
}}
#blue-btn>button {{
  background: linear-gradient(90deg, #007bff, #66b2ff);
}}
#green-btn>button {{
  background: linear-gradient(90deg, #00cc44, #80ff80);
}}
#white-btn>button {{
  background: linear-gradient(90deg, #cccccc, #ffffff);
  color: black;
}}
</style>
""", unsafe_allow_html=True)

# --- Layout: Left = Buttons, Right = Video Feed ---
left_col, right_col = st.columns([1, 3])

with left_col:
    st.markdown("## 🎛️ Select Color to Detect")
    st.markdown("Click a button below to start detection:")

    red_clicked = st.container()
    with red_clicked:
        red_pressed = st.button("Detect Red", key="red-btn")

    blue_clicked = st.container()
    with blue_clicked:
        blue_pressed = st.button("Detect Blue", key="blue-btn")

    green_clicked = st.container()
    with green_clicked:
        green_pressed = st.button("Detect Green", key="green-btn")

    white_clicked = st.container()
    with white_clicked:
        white_pressed = st.button("Detect White", key="white-btn")

    st.markdown("---")
    stop = st.button("⛔ Stop Detection")

with right_col:
    st.markdown("## 🎥 Live Color Detection Feed")
    frame_placeholder = st.empty()

# --- Helper Function for Detection ---
def detect_color(low, high, color_name):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("⚠️ Could not open webcam.")
        return

    st.info(f"Detecting {color_name} color. Press 'Stop Detection' to end.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, low, high)
        result = cv2.bitwise_and(frame, frame, mask=mask)
        rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(rgb, channels="RGB")

        if stop:
            break

    cap.release()
    st.success(f"✅ {color_name} Detection Stopped.")

# --- Button Logic ---
if red_pressed:
    detect_color(np.array([161, 155, 84]), np.array([179, 255, 255]), "Red")

elif blue_pressed:
    detect_color(np.array([94, 80, 2]), np.array([126, 255, 255]), "Blue")

elif green_pressed:
    detect_color(np.array([40, 100, 100]), np.array([102, 255, 255]), "Green")

elif white_pressed:
    detect_color(np.array([0, 0, 200]), np.array([179, 40, 255]), "White")
    
    