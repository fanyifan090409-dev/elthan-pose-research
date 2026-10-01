from ultralytics import YOLO

WT = "/home/ubuntu/isef/runs/pose-round3/weights/best.pt"

model = YOLO(WT)
print("=== ONNX ===")
model.export(format="onnx", imgsz=640, half=False, opset=12)
print("=== DONE ===")
