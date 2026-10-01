# -*- coding: utf-8 -*-
# PT best.pt 走与 NCNN 完全相同的 640 固定左上角 letterbox 协议，
# 官方 torch decode + NMS，输出 native 空间 pred JSON 供 eval_ncnn_ap3 对拍。
# 用途：区分「固定640 vs rect 画布」协议差 与 NCNN 导出/量化损失。
import json
import os
import time

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.utils.nms import non_max_suppression

ROOT = "/home/ubuntu/isef"
CKPT = f"{ROOT}/runs/pose-round5-3/weights/best.pt"
VAL_IMG = f"{ROOT}/datasets/pose-round3/images/val"
OUT = f"{ROOT}/deploy/ncnn/pose-round5-3/ap-eval"
IMGSZ = 640
CONF = 0.001
IOU = 0.7
MAXDET = 300
BATCH = 32


def letterbox(im):
    h, w = im.shape[:2]
    r = min(IMGSZ / h, IMGSZ / w)
    uw, uh = int(round(w * r)), int(round(h * r))
    canvas = np.full((IMGSZ, IMGSZ, 3), 114, dtype=np.uint8)
    canvas[:uh, :uw] = cv2.resize(im, (uw, uh), interpolation=cv2.INTER_LINEAR)
    return canvas, uw / w


gt = json.load(open(f"{OUT}/gt.json"))
imgs = gt["images"]
model = YOLO(CKPT).model.eval().cuda()

preds = []
t0 = time.time()
for b0 in range(0, len(imgs), BATCH):
    chunk = imgs[b0:b0 + BATCH]
    xs, ratios, ws, hs = [], [], [], []
    for im in chunk:
        im0 = cv2.imread(os.path.join(VAL_IMG, im["file_name"]))
        canvas, ratio = letterbox(im0)
        xs.append(canvas[:, :, ::-1].transpose(2, 0, 1))
        ratios.append(ratio)
        ws.append(im["width"])
        hs.append(im["height"])
    x = torch.from_numpy(np.ascontiguousarray(np.stack(xs))).float().div(255).cuda()
    with torch.no_grad():
        y = model(x)
    if isinstance(y, (list, tuple)):
        y = y[0]
    rs = non_max_suppression(y, CONF, IOU, nc=7, max_det=MAXDET)
    for j, r in enumerate(rs):
        ratio = ratios[j]
        w, h = ws[j], hs[j]
        for det in r.tolist():
            x1, y1, x2, y2, sc, cls = det[:6]
            kpt = np.array(det[6:]).reshape(-1, 3)
            x1, y1, x2, y2 = x1 / ratio, y1 / ratio, x2 / ratio, y2 / ratio
            kpt[:, :2] /= ratio
            bx = max(0.0, min(x1, w)); by = max(0.0, min(y1, h))
            bw = min(x2 - x1, w - bx); bh = min(y2 - y1, h - by)
            if bw <= 0 or bh <= 0:
                continue
            preds.append({
                "image_id": chunk[j]["id"],
                "category_id": int(cls) + 1,
                "bbox": [bx, by, bw, bh],
                "area": bw * bh,
                "score": sc,
                "keypoints": kpt.flatten().tolist(),
            })
    if (b0 + BATCH) % 512 == 0 or b0 + BATCH >= len(imgs):
        print(f"{min(b0 + BATCH, len(imgs))}/{len(imgs)}  {time.time() - t0:.0f}s", flush=True)

json.dump(preds, open(f"{OUT}/pred_pt640.json", "w"))
print(f"DONE {len(preds)} preds in {time.time() - t0:.0f}s")
