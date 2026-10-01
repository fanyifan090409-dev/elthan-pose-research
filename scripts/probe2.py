# -*- coding: utf-8 -*-
# 插桩 preprocess：打印 m.val() 真实 batch 的 img 形状 / ori_shape / ratio_pad 后退出
from ultralytics import YOLO
from ultralytics.models.yolo.pose import PoseValidator

BEST = "/home/ubuntu/isef/runs/pose-round5-3/weights/best.pt"
DATA = "/home/ubuntu/isef/datasets/pose-round3/data.yaml"

orig_pre = PoseValidator.preprocess


def diag_pre(self, batch):
    r = orig_pre(self, batch)
    print("IMG shape:", tuple(batch["img"].shape))
    for i in range(min(5, len(batch["ori_shape"]))):
        print(i, "ori", tuple(batch["ori_shape"][i]), "ratio_pad", batch["ratio_pad"][i])
    print("batch keys:", sorted(batch.keys()))
    raise SystemExit(0)


PoseValidator.preprocess = diag_pre

m = YOLO(BEST)
m.val(data=DATA, imgsz=640, batch=64, device=0,
      project="/home/ubuntu/isef/runs/repro", name="probe_batch")
