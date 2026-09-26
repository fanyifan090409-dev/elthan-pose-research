#!/usr/bin/env python3
"""train_pose_round3.py — YOLO11s-pose round-3 微调：全量 COCO-Pose + MPII，双卡 DDP"""
from ultralytics import YOLO

model = YOLO("/home/ubuntu/isef/weights/yolo11s-pose.pt")
model.train(
    data="/home/ubuntu/isef/datasets/pose-round3/data.yaml",
    epochs=60,
    imgsz=640,
    batch=64,
    device="0,1",
    project="/home/ubuntu/isef/runs",
    name="pose-round3",
    patience=15,
    close_mosaic=10,
    workers=8,
    seed=42,
    plots=True,
    cache=False,
)
