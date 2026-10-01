# -*- coding: utf-8 -*-
# round5-3 端侧链第 1 步：best.pt -> best.onnx（opset 12, FP32, imgsz 640）
from ultralytics import YOLO

WT = "/home/ubuntu/isef/runs/pose-round5-3/weights/best.pt"

model = YOLO(WT)
print("=== ONNX ===")
model.export(format="onnx", imgsz=640, half=False, opset=12, device=0)
print("=== DONE ===")
