# 导出模型
from ultralytics import YOLO
# Load a model
model = YOLO("/home/mjy/ultralytics3/runs/detect/train2/weights/best.pt")
# model.export(format='engine',int8=True,dynamic=True,batch=16,data='/home/mjy/ultralytics/data/drone3.yaml')
# model.export(format='engine',half=True)
# model.export(format='engine')
model.export(format='onnx')
