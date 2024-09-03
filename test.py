# 训练
from ultralytics import YOLO
model = YOLO('/home/mjy/ultralytics3/runs/detect/train2/weights/best.pt')  # build from YAML and transfer weights
model.val(data='/home/mjy/ultralytics3/vedai.yaml',split='test',batch=16,imgsz=640)
# metrics = model.val(data='/home/mjy/ultralytics/data/drone2.yaml',split='test',imgsz=640,batch=16)