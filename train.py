# 训练
from ultralytics import YOLO
model = YOLO('./yolov8n.yaml')  # build from YAML and transfer weights
model.train(data='/home/mjy/ultralytics3/vedai.yaml',batch=16,imgsz=640)
