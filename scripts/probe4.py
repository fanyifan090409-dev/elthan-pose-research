# -*- coding: utf-8 -*-
# 决定性实验：同一张 (169,640) 图，两种输入（居中pad vs 拉伸），看哪种复现官方 PD
import cv2
import numpy as np
import torch

from ultralytics import YOLO
from ultralytics.utils.nms import non_max_suppression

BEST = "/home/ubuntu/isef/runs/pose-round5-3/weights/best.pt"
IMG = "/home/ubuntu/isef/datasets/pose-round3/images/val/000000000139.jpg"

m = YOLO(BEST)
model = m.model.eval()
im = cv2.imread(IMG)
print("ori:", im.shape)

canvas = np.full((384, 672, 3), 114, np.uint8)
canvas[107:107 + 169, 16:16 + 640] = im
im_stretch = cv2.resize(im, (672, 384))

for name, arr in [("padded", canvas), ("stretched", im_stretch)]:
    x = torch.from_numpy(arr[:, :, ::-1].transpose(2, 0, 1).copy()).float().div(255).unsqueeze(0).cuda()
    with torch.no_grad():
        pred = model(x)[0]
    r = non_max_suppression(pred, 0.001, 0.7, nc=1, max_det=300)[0]
    print(name, "boxes:", [tuple(round(v, 1) for v in b[:4]) for b in r[:3].tolist()])
