from ultralytics import YOLO

#load a model
model = YOLO("Autonomous/yolo_weights/yolov8n.pt")   #build a new model from scratch

#use a model
results = model.train(data="config.yaml", epochs = 1)   #train the model

