#!/usr/bin/env python3
"""train_pose_round1.py — YOLO11n-pose round-1 微调：COCO-Pose 15k + MPII（task #21）"""
from ultralytics import YOLO

model = YOLO("/home/ubuntu/isef/weights/yolo11n-pose.pt")
model.train(
    data="/home/ubuntu/isef/datasets/pose-round1/data.yaml",
    epochs=60,
    imgsz=640,
    batch=64,
    device=0,
    project="/home/ubuntu/isef/runs",
    name="pose-round1",
    patience=15,
    close_mosaic=10,
    workers=8,
    seed=42,
    plots=True,
    cache=False,
)
