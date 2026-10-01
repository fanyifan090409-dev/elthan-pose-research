# -*- coding: utf-8 -*-
# 决定性实验 v3：官方 val 批次0 —— 居中pad输入 vs 拉伸输入，输出框对比 + 可视化
import cv2
import numpy as np
import torch

from ultralytics import YOLO
from ultralytics.models.yolo.pose import PoseValidator
from ultralytics.utils.nms import non_max_suppression

CKPT = "/home/ubuntu/isef/runs/pose-round5-3/weights/best.pt"
DATA = "/home/ubuntu/isef/datasets/pose-round3/data.yaml"

m = YOLO(CKPT)
model = m.model.eval().cuda()

orig_pre = PoseValidator.preprocess


def hooked_pre(self, batch):
    orig_pre(self, batch)
    img = batch["img"]  # (64,3,H,W) cuda
    H, W = img.shape[2:]
    print("batch img:", tuple(img.shape))
    print("im_file[0]:", batch["im_file"][0])
    print("ori[0]:", tuple(batch["ori_shape"][0]), "resized[0]:", tuple(batch["resized_shape"][0]))
    print("ratio_pad[0]:", batch["ratio_pad"][0])

    # 分支1：官方输入（居中pad后的 canvas）直接喂模型
    with torch.no_grad():
        pred = model(img[0:1])[0]
    r_pad = non_max_suppression(pred, 0.001, 0.7, nc=1, max_det=300)[0]
    print("padded   PD:", [tuple(round(v, 1) for v in b[:4]) for b in r_pad[:3].tolist()])

    # 分支2：原始图按加载比还原内容区，拉伸到 canvas 尺寸再喂
    raw = cv2.imread(batch["im_file"][0])
    oh, ow = raw.shape[:2]
    rh, rw = batch["resized_shape"][0]
    content = cv2.resize(raw, (rw, rh))  # 与 load_image 相同的等比缩放
    stretched = cv2.resize(content, (W, H))
    x2 = torch.from_numpy(stretched[:, :, ::-1].transpose(2, 0, 1).copy()).float().div(255).unsqueeze(0).cuda()
    with torch.no_grad():
        pred2 = model(x2)[0]
    r_str = non_max_suppression(pred2, 0.001, 0.7, nc=1, max_det=300)[0]
    print("stretched PD:", [tuple(round(v, 1) for v in b[:4]) for b in r_str[:3].tolist()])

    # 可视化：padded canvas 上画 GT（绿，norm xywh × canvas）+ 两分支 PD（蓝=pad，红=stretch）
    arr = (img[0].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8).copy()
    m0 = batch["batch_idx"].cpu().numpy().squeeze() == 0
    for x, y, w, h in batch["bboxes"].cpu().numpy()[m0]:
        x1, y1, x2, y2 = x * W, y * H, (x + w) * W, (y + h) * H
        cv2.rectangle(arr, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(arr, "GT", (int(x1), max(int(y1) - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    for b in r_pad[:3].tolist():
        cv2.rectangle(arr, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 0, 0), 2)
    for b in r_str[:3].tolist():
        cv2.rectangle(arr, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 0, 255), 2)
    cv2.imwrite("/tmp/p5_padded_gt.jpg", arr)
    cv2.imwrite("/tmp/p5_raw.jpg", raw)
    print("saved /tmp/p5_padded_gt.jpg /tmp/p5_raw.jpg  raw shape:", raw.shape)
    raise SystemExit(0)


PoseValidator.preprocess = hooked_pre
m.val(data=DATA, imgsz=640, batch=64, device=0, project="/home/ubuntu/isef/runs/probe", name="p5")
