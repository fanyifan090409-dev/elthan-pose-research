from ultralytics import YOLO

m = YOLO("/home/ubuntu/isef/runs/pose-round5-2/weights/last.pt")
m.train(data="/home/ubuntu/isef/datasets/pose-round5-ablation/data.yaml",
        epochs=30, batch=128, imgsz=640, device="0,1",
        lr0=0.0025, patience=30, close_mosaic=10, seed=42,
        project="/home/ubuntu/isef/runs", name="pose-round5-abl-nocrowd")
