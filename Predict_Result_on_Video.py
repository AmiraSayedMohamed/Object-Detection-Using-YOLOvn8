import os
from ultralytics import YOLO
import cv2

video_path = os.path.join(r"C:\Users\PC\Desktop\Final_Data\video.wmv")

# Get the absolute path to the model file
model_path = os.path.abspath(r'C:\Users\PC\Downloads\weights\weights\best.pt')
# load the model
model = YOLO(model_path)

# Perform object detection
detection_output = model.predict(source=video_path, conf=0.25, save=True)
