# -*- coding: utf-8 -*-
# dump batch[0] 图像 + 定位内容区域 + GT bbox 画布坐标，验证真实输入空间
import numpy as np
from ultralytics import YOLO
from ultralytics.models.yolo.pose import PoseValidator

BEST = "/home/ubuntu/isef/runs/pose-round5-3/weights/best.pt"
DATA = "/home/ubuntu/isef/datasets/pose-round3/data.yaml"

orig_pre = PoseValidator.preprocess


def diag_pre(self, batch):
    r = orig_pre(self, batch)
    img = batch["img"][0].permute(1, 2, 0).cpu().numpy() * 255
    img = img.astype(np.uint8)
    mask = ~((img == 114).all(-1))
    ys, xs = np.where(mask)
    print("canvas:", img.shape, "content rows", int(ys.min()), "-", int(ys.max()),
          "cols", int(xs.min()), "-", int(xs.max()))
    idx = batch["batch_idx"] == 0
    bb = batch["bboxes"][idx].cpu().numpy()
    kp = batch["keypoints"][idx].cpu().numpy()
    H, W = batch["img"].shape[2:]
    print("GT bbox norm:", bb[0].round(4), "-> canvas xyxy:",
          (np.concatenate([bb[0, :2], bb[0, :2] + bb[0, 2:]]) * np.array([W, H, W, H])).round(1))
    print("GT kpt0 canvas:", (kp[0, 0, :2] * np.array([W, H])).round(1))
    import cv2
    cv2.imwrite("/tmp/img0.png", img[:, :, ::-1])
    raise SystemExit(0)


PoseValidator.preprocess = diag_pre

m = YOLO(BEST)
m.val(data=DATA, imgsz=640, batch=64, device=0,
      project="/home/ubuntu/isef/runs/repro", name="probe_batch")
