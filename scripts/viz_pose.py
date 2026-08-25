#!/usr/bin/env python3
"""ISEF pose visualization: render skeleton overlays from YOLO26-pose predictions."""
import sys, os, time
import numpy as np
from ultralytics import YOLO

WEIGHTS = sys.argv[1] if len(sys.argv) > 1 else "/home/ubuntu/isef/weights/yolo26s-pose.pt"
SRC = sys.argv[2] if len(sys.argv) > 2 else "/home/ubuntu/isef/data/viz_src"
OUT = "/home/ubuntu/isef/docs/viz"
os.makedirs(OUT, exist_ok=True)

model = YOLO(WEIGHTS)
model.to("cuda:1")

# gather source images
imgs = []
for root, _, files in os.walk(SRC):
    for f in files:
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            imgs.append(os.path.join(root, f))
imgs = sorted(imgs)[:8]
if not imgs:
    # fallback: ultralytics bundled asset
    import ultralytics
    p = os.path.join(os.path.dirname(ultralytics.__file__), "assets", "bus.jpg")
    imgs = [p]
print("VIZ sources:", len(imgs), "images")

t0 = time.time()
results = model.predict(imgs, imgsz=640, conf=0.25, device=1, verbose=False)
for r, src in zip(results, imgs):
    im = r.plot(line_width=2, labels=True)  # BGR ndarray, kpts auto-rendered
    name = os.path.splitext(os.path.basename(src))[0] + "_pred.jpg"
    import cv2
    cv2.imwrite(os.path.join(OUT, name), im)
    k = r.keypoints
    shape = None if k is None else tuple(k.data.shape)
    print(f"saved {name} boxes={len(r.boxes) if r.boxes is not None else 0} kpts={shape}")
print(f"VIZ_DONE in {time.time()-t0:.1f}s -> {OUT}")
