import cv2
from ultralytics import YOLO
import numpy as np

# Double-check and adjust the video path if necessary
video_path = "peCa.mp4"

cap = cv2.VideoCapture(video_path)

model = YOLO(r"C:\Users\PC\Downloads\weights\weights\best.pt")

while True:
    try:
        # Check for successful frame capture before processing
        ret, frame = cap.read()
        if not ret or frame is None:
            print("No frame captured. Exiting...")
            break

        results = model(frame, show=True, conf=0.3)
        result = results[0]

        bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
        confidences = np.array(result.boxes.conf.cpu(), dtype="float")
        classes = np.array(result.boxes.cls.cpu(), dtype="int")

        for bbox, confi, cls in zip(bboxes, confidences, classes):
            (x, y, x2, y2) = bbox
            class_id = int(cls)
            object_name = model.names[class_id]

            cv2.rectangle(frame, (x, y), (x2, y2), (0, 0, 225), 2)
            cv2.putText(frame, f"{object_name} {confi:.2f}", (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 225), 2)
            cx = (int(x + x2) // 2)
            cy = (int(y + y2) // 2)
            cv2.circle(frame, (cx, cy), 10, (0, 0, 255), -1)

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1)
        if key == 27:
            break
    except Exception as e:
        print(f"An error occurred: {e}")

cap.release()
cv2.destroyAllWindows()
