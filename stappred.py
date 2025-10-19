import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="Color Detection", layout="centered")

# --- Styling (Orange gradient with AI background) ---
#DEFAULT_BG = "https://images.unsplash.com/photo-1508921912186-1d1a45ebb3c1?auto=format&fit=crop&w=1400&q=80"
DEFAULT_BG="https://64.media.tumblr.com/911052856c0d258c45ef61c36379ae29/2dade201121f6c1f-88/s1280x1920/5bce19b73bd62da18c2e139637d7b3cf6eee093d.jpg"
#DEFAULT_BG ="c:\users\ttwrd\downloads\face2.jpg"
st.markdown(f"""
<style>
[data-testid="stAppViewContainer"] {{
  background-image: linear-gradient(135deg, #ff8a00 0%, #ffd59a 100%), url('{DEFAULT_BG}');
  background-size: cover;
  background-position: center;
  background-repeat: no-repeat;
  background-blend-mode: overlay;
}}
[data-testid="stHeader"] {{background: rgba(0,0,0,0);}}
.stButton>button {{
  background: linear-gradient(90deg,#ff7a18,#ffcc80);
  color: white; border: none; padding: 10px 18px;
  border-radius: 12px; box-shadow: 0 6px 18px rgba(0,0,0,0.18);
  font-weight: 600;
}}
.stTextInput>div>div>input {{
  border-radius: 12px; padding: 12px 14px;
  border: 1px solid rgba(0,0,0,0.12);
  box-shadow: 0 4px 12px rgba(0,0,0,0.06) inset;
}}
</style>
""", unsafe_allow_html=True)

st.title("🎨 Red Color Detection with Streamlit + OpenCV")

# Textbox
user_text = st.text_input("Enter text here:", placeholder="Type anything...")

# Columns for buttons
col1, col2, col3, col4 = st.columns(4)


# --- Button 1: Run Color Detection ---
if col1.button("Start Red Detection"):
    st.write("🕹️ Running live red color detection...")



    cap = cv2.VideoCapture(0)

    while True:
        _, frame = cap.read()
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Red color
        low_red = np.array([161, 155, 84]) # lowest hue would be - 161,155,84( how do i found this i tested before and found this) 
        high_red = np.array([179, 255, 255])
        #mask = cv2.inRange(hsv_frame, low_red, high_red) 
            
        red_mask = cv2.inRange(hsv_frame, low_red, high_red) #we create maskk on hsv frame and then low red or high red
        red = cv2.bitwise_and(frame, frame, mask=red_mask)


        cv2.imshow("Frame", frame) 
        #cv2.imshow('Red mask', mask) 
        cv2.imshow('Red', red)
        
        
        key = cv2.waitKey(1)
        if key ==27:
            break



        st.success("✅ Detection stopped.")
        
elif col2.button("Blue color"):
     st.info("Button 2 clicked")
     cap = cv2.VideoCapture(0)
     while True:
        _, frame = cap.read()
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    
      # Blue color
        low_blue = np.array([94, 80, 2])
        high_blue = np.array([126, 255, 255])
        blue_mask = cv2.inRange(hsv_frame, low_blue, high_blue)
        blue = cv2.bitwise_and(frame, frame, mask=blue_mask)
        cv2.imshow("Frame", frame) 
          #cv2.imshow('Red mask', blue_mask) 
        cv2.imshow('Red', red)
        key = cv2.waitKey(1)
        if key ==27:
              break

elif col3.button("Start Green Detection"):
     st.warning("Button 3 clicked")
     # Green color
     cap = cv2.VideoCapture(0)
     while True:
        _, frame = cap.read()
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    
        low_green = np.array([40, 100, 100])
        high_green = np.array([102, 255,    255])
        green_mask = cv2.inRange(hsv_frame, low_green, high_green)
        green = cv2.bitwise_and(frame, frame, mask=green_mask)
        cv2.imshow("Frame", frame) 
          
        cv2.imshow('Green', green)
        key = cv2.waitKey(1)
        if key ==27:
                break    

elif col4.button("Start White onlyDetection"):
     st.error("Button 4 clicked")
     cap = cv2.VideoCapture(0)
     while True:
        _, frame = cap.read()
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    
        low = np.array([0, 42, 0])
        high = np.array([179, 255, 255])
        mask = cv2.inRange(hsv_frame, low, high)
        result = cv2.bitwise_and(frame, frame, mask=mask)
        cv2.imshow("Frame", frame) 
          
        cv2.imshow('Except white', result)
        key = cv2.waitKey(1)
        if key ==27:
                break  

st.markdown("---")
st.caption("Built with ❤️ using Streamlit and OpenCV")