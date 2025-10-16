# --------------------------------------------------------
# 3D Object Detection from Video using MediaPipe Objectron
# Compatible with Python 3.11.13 + Windows
# --------------------------------------------------------

import cv2
import mediapipe as mp
import os

# ----> Check your video path here
VIDEO_PATH = r"C:\Users\ttwrd\Downloads\shoes.mp4"

# ----> Verify that file exists
if not os.path.exists(VIDEO_PATH):
    raise FileNotFoundError(f"Video not found at: {VIDEO_PATH}")

# ----> Try to open video with OpenCV
cap = cv2.VideoCapture(VIDEO_PATH, cv2.CAP_FFMPEG)
if not cap.isOpened():
    print("⚠️ OpenCV failed to open the video. Trying imageio fallback...")

    import imageio.v3 as iio
    import numpy as np

    frames = iio.imiter(VIDEO_PATH)
    mp_objectron = mp.solutions.objectron
    mp_drawing = mp.solutions.drawing_utils

    with mp_objectron.Objectron(
        static_image_mode=False,
        max_num_objects=5,
        min_detection_confidence=0.4,
        min_tracking_confidence=0.7,
        model_name='Shoe'  # Try 'Cup', 'Chair', or 'Camera' if needed
    ) as objectron:
        for frame in frames:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            results = objectron.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

            if results.detected_objects:
                for obj in results.detected_objects:
                    mp_drawing.draw_landmarks(
                        frame_bgr, obj.landmarks_2d, mp_objectron.BOX_CONNECTIONS)
                    mp_drawing.draw_axis(frame_bgr, obj.rotation, obj.translation)

            cv2.imshow("MediaPipe Objectron (imageio)", cv2.flip(frame_bgr, 1))
            if cv2.waitKey(20) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
            
