#!/usr/bin/env python3
"""train_pose_round5.py — YOLO11s-pose round-5：round4 best.pt 起步，+OCHuman 2,837 + OTP 4,300（控比例），batch 64→128

注意：ultralytics DDP 下 batch // world_size = 每卡量。128 = 64/卡（round4 为 32/卡）。
若 OOM，新版会自动减半重试；仍失败则手动降 96。
"""
from ultralytics import YOLO

model = YOLO("/home/ubuntu/isef/runs/pose-round4/weights/best.pt")
model.train(
    data="/home/ubuntu/isef/datasets/pose-round5/data.yaml",
    epochs=60,
    imgsz=640,
    batch=128,
    device="0,1",
    project="/home/ubuntu/isef/runs",
    name="pose-round5",
    patience=15,
    close_mosaic=10,
    workers=8,
    seed=42,
    plots=True,
    cache=False,
)
