#!/usr/bin/env python3
"""train_pose_round5_resume.py — round5 续训：last.pt(epoch16) 起步，lr0 0.005（取消 batch128 的自动翻倍），patience 30

背景：首轮 batch=128 触发 nbs=64 自动 lr×2，lr 峰值验证崩至 0.344，epoch 16 恢复到 0.472 时被 patience=15 早停。
本轮 lr0 回到 round4 等效值（0.01×64/64=0.01 → 这里显式 0.005 = 0.01×64/128），测试「batch 128 + 正确 lr」。
"""
from ultralytics import YOLO

model = YOLO("/home/ubuntu/isef/runs/pose-round5/weights/last.pt")
model.train(
    resume=True,
    data="/home/ubuntu/isef/datasets/pose-round5/data.yaml",
    epochs=45,
    imgsz=640,
    batch=128,
    device="0,1",
    project="/home/ubuntu/isef/runs",
    name="pose-round5",
    patience=30,
    lr0=0.005,
    close_mosaic=10,
    workers=8,
    seed=42,
    plots=True,
    cache=False,
)
