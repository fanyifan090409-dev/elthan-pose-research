"""pose-round5 续训：round5-2 last.pt 起 +30 epoch。

- lr0=0.0025（nbs 缩放后等效 0.005，为 run2 峰值一半）：
  run2 结束时 lr 已余弦衰减至 ~0.001，全档重启会重演 run1/run2 的
  epoch1-2 回落坑（0.49→0.46），半档重启减少浪费。
- 数据/val 与 round5 完全一致，保证曲线可比。
"""
from ultralytics import YOLO

model = YOLO("/home/ubuntu/isef/runs/pose-round5-2/weights/last.pt")
model.train(
    data="/home/ubuntu/isef/datasets/pose-round5/data.yaml",
    epochs=30,
    batch=128,
    imgsz=640,
    device="0,1",
    workers=8,
    patience=30,
    lr0=0.0025,
    close_mosaic=10,
    seed=42,
    name="pose-round5-3",
    project="/home/ubuntu/isef/runs",
)
