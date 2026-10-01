from ultralytics import YOLO

m = YOLO("/home/ubuntu/isef/runs/pose-round5-2/weights/last.pt")
m.train(data="/home/ubuntu/isef/datasets/pose-round5-ablationC/data.yaml",
        epochs=30, batch=96, imgsz=640, device=1,
        lr0=0.00333, patience=30, close_mosaic=10, seed=42,
        project="/home/ubuntu/isef/runs", name="pose-round5-abl-nootp")
