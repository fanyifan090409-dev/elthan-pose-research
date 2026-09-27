#!/usr/bin/env python3
"""train_pose_round4.py — YOLO11s-pose round-4：round3 best.pt 起步，混入筛选后 CrowdPose（task #103）"""
from ultralytics import YOLO

model = YOLO("/home/ubuntu/isef/runs/pose-round3/weights/best.pt")
model.train(
    data="/home/ubuntu/isef/datasets/pose-round4/data.yaml",
    epochs=60,
    imgsz=640,
    batch=64,
    device="0,1",
    project="/home/ubuntu/isef/runs",
    name="pose-round4",
    patience=15,
    close_mosaic=10,
    workers=8,
    seed=42,
    plots=True,
    cache=False,
)
