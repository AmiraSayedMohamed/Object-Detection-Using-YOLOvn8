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

        results = model(frame,show=True, conf=0.3)
        result = results[0]
        print(result)
        bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
        classes = np.array(result.boxes.cls.cpu(), dtype="int")

        for result in results:
            if result.boxes:
                box = result.boxes[0]
                class_id = int(box.cls)
                confi= float(box.conf)
                print('confidence is ', confi)
                object_name = model.names[class_id]


        for cls, bbox in zip(classes, bboxes):
            (x, y, x2, y2) = bbox
            cv2.rectangle(frame, (x, y), (x2, y2), (0, 0, 225), 2)
            cv2.putText(frame, f"{object_name} {confi:0.2}" , (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 225), 5)
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
