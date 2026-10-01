# -*- coding: utf-8 -*-
# 探测 m.val() 真实比较空间：rect 是否生效、batch img 形状、ratio_pad
import glob
import sys

from ultralytics.models.yolo.pose import PoseValidator

BEST = "/home/ubuntu/isef/runs/pose-round5-3/weights/best.pt"
DATA = "/home/ubuntu/isef/datasets/pose-round3/data.yaml"
print("best:", BEST)

v = PoseValidator(args=dict(model=BEST, data=DATA, imgsz=640, batch=16))
print("args.rect =", v.args.rect)
print("args.imgsz =", v.args.imgsz)
dl = v.get_dataloader(v.data["val"], 16)
print("dataloader batches:", len(dl), "dataset imgs:", len(dl.dataset))
batch = next(iter(dl))
print("batch img shape:", tuple(batch["img"].shape))
for i in range(min(4, batch["img"].shape[0])):
    print(f"  si={i} ori_shape={tuple(batch['ori_shape'][i].tolist())} ratio_pad={batch['ratio_pad'][i].tolist()}")
