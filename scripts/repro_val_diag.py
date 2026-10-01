# -*- coding: utf-8 -*-
# 复现官方 val + 诊断前 3 张图的真实比较空间：
# GT bbox/kpt（_prepare_batch 之后）vs pred bbox/kpt，打印 box_iou / kpt_iou 首行
from ultralytics import YOLO
from ultralytics.models.yolo.pose import PoseValidator
from ultralytics.utils.metrics import box_iou, kpt_iou

CKPT = "/home/ubuntu/isef/runs/pose-round5-3/weights/best.pt"
DATA = "/home/ubuntu/isef/datasets/pose-round3/data.yaml"

orig_pb = PoseValidator._process_batch


def diag_pb(self, preds, batch):
    r = orig_pb(self, preds, batch)
    if self.seen <= 3 and batch["cls"].shape[0] and preds["cls"].shape[0]:
        gb, pbx = batch["bboxes"], preds["bboxes"]
        iou = box_iou(gb, pbx)
        area = (gb[:, 2] - gb[:, 0]) * (gb[:, 3] - gb[:, 1]) * 0.53
        kio = kpt_iou(batch["keypoints"], preds["keypoints"], area, self.sigma)
        print(f"--- image {self.seen} ori={tuple(batch['ori_shape'])} imgsz={tuple(batch['imgsz'])}")
        print("GT box[0]:", [round(v, 1) for v in gb[0].tolist()])
        print("GT kpt[0,:3]:", batch["keypoints"][0, :3, :2].flatten().tolist())
        print("PD box[0]:", [round(v, 1) for v in pbx[0].tolist()])
        print("PD kpt[0,:3]:", preds["keypoints"][0, :3, :2].flatten().tolist())
        print("box_iou row0 max:", round(iou[0].max().item(), 4), " kpt_iou row0 max:", round(kio[0].max().item(), 4))
        print("tp_p row0:", [int(x) for x in r["tp_p"][0].tolist()])
    return r


PoseValidator._process_batch = diag_pb

m = YOLO(CKPT)
res = m.val(data=DATA, imgsz=640, batch=64, device=0,
            project="/home/ubuntu/isef/runs/repro", name="round53_val_diag")
print("RESULT box:", res.box.map50, res.box.map, "| pose:", res.pose.map50, res.pose.map)
