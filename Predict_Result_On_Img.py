import torch
import os
import cv2
from ultralytics import YOLO
import numpy

# Get the absolute path to the model file
model_path = os.path.abspath(r'C:\Users\PC\Downloads\weights\weights\best.pt')
model = YOLO(model_path)

img_path = r"C:\Users\PC\Desktop\Final_Data\Version1\train\images\Screenshot-from-2024-04-20-01-40-49_png.rf.216470630fcd6e1f545c59c9044d4b00.jpg"
detection_output = model.predict(source=img_path, conf=0.25, save=True)

#Display Tensor array
print(detection_output)

#Display numpy array
print(detection_output[0].numpy())


cv2.waitKey(0)
cv2.destroyAllWindows()
