from ultralytics import YOLO

m = YOLO("yolo11s-pose.pt")
m.train(data="/home/ubuntu/isef/datasets/overfit-test/data.yaml",
        epochs=100, batch=16, imgsz=640, device=0,
        lr0=0.01, nbs=16, close_mosaic=0, patience=100,
        project="/home/ubuntu/isef/runs", name="overfit-test",
        seed=42, workers=2)
