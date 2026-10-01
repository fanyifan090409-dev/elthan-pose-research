# -*- coding: utf-8 -*-
# round5-3 INT8 AP 评测 v2：对齐 ultralytics v8.4 PoseValidator 官方口径
# 官方协议：全部在 letterbox 640 空间对拍；pose 用 kpt_iou(OKS_SIGMA, area=GT框面积*0.53)
# + match_predictions(iouv 0.5:0.95) + ap_per_class；NMS conf=0.001 iou=0.7 max_det=300
# 输入：run1 存的 pred_{tag}.json（原生坐标 NMS 结果）+ gt.json，乘回 letterbox 比例重算
import json
import numpy as np
import torch
from ultralytics.utils.metrics import OKS_SIGMA, ap_per_class, kpt_iou

OUT = "/home/ubuntu/isef/deploy/ncnn/pose-round5-3/ap-eval"
IOUV = np.linspace(0.5, 0.95, 10)


def letterbox_ratio(w, h):
    r = min(640 / h, 640 / w)
    return int(round(w * r)) / w


def match_predictions(pred_classes, true_classes, iou):
    correct = np.zeros((pred_classes.shape[0], IOUV.shape[0])).astype(bool)
    correct_class = true_classes[:, None] == pred_classes
    iou = iou * correct_class
    for i, threshold in enumerate(IOUV):
        matches = np.array(np.nonzero(iou >= threshold)).T
        if matches.shape[0]:
            if matches.shape[0] > 1:
                matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return correct


def box_iou_np(gt, pd):
    x1 = np.maximum(gt[:, None, 0], pd[None, :, 0])
    y1 = np.maximum(gt[:, None, 1], pd[None, :, 1])
    x2 = np.minimum(gt[:, None, 2], pd[None, :, 2])
    y2 = np.minimum(gt[:, None, 3], pd[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    ga = (gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1])
    pa = (pd[:, 2] - pd[:, 0]) * (pd[:, 3] - pd[:, 1])
    return inter / (ga[:, None] + pa[None, :] - inter + 1e-16)


def evaluate(tag):
    gt = json.load(open(f"{OUT}/gt.json"))
    preds = json.load(open(f"{OUT}/pred_{tag}.json"))
    imw = {im["id"]: im["width"] for im in gt["images"]}
    imh = {im["id"]: im["height"] for im in gt["images"]}

    # GT 按官方口径：归一化坐标 × 640（两轴都乘 640，即拉伸到正方形），
    # 而预测保持在 letterbox 640 内容空间 —— 与 v8.4 _prepare_batch 完全一致
    gts = {}
    for a in gt["annotations"]:
        iid = a["image_id"]
        sx, sy = 640 / imw[iid], 640 / imh[iid]
        x, y, w, h = a["bbox"]
        k = np.array(a["keypoints"]).reshape(-1, 3)
        k[:, 0] *= sx
        k[:, 1] *= sy
        gts.setdefault(iid, []).append({
            "cls": a["category_id"] - 1,
            "box": np.array([x * sx, y * sy, (x + w) * sx, (y + h) * sy]),
            "wh": np.array([w * sx, h * sy]),
            "kpt": k,
        })

    preds_by_img = {}
    for p in preds:
        preds_by_img.setdefault(p["image_id"], []).append(p)

    tp_l, tp_p_l, conf_l, pcls_l, tcls_l = [], [], [], [], []
    for iid, g in gts.items():
        gcls = np.array([a["cls"] for a in g])
        gbox = np.stack([a["box"] for a in g])
        gwh = np.stack([a["wh"] for a in g])
        gk = np.stack([a["kpt"] for a in g])
        tcls_l.append(gcls)
        ps = preds_by_img.get(iid, [])
        if not ps:
            continue
        r = letterbox_ratio(imw[iid], imh[iid])
        pbox = np.stack([np.array([p["bbox"][0], p["bbox"][1],
                                   p["bbox"][0] + p["bbox"][2],
                                   p["bbox"][1] + p["bbox"][3]]) * r for p in ps])
        pk = np.stack([np.array(p["keypoints"]).reshape(-1, 3) for p in ps])
        pk[:, :, :2] *= r
        pcls = np.array([p["category_id"] - 1 for p in ps])
        pscores = np.array([p["score"] for p in ps])

        tp = match_predictions(pcls, gcls, box_iou_np(gbox, pbox))
        gt_area = gwh[:, 0] * gwh[:, 1] * 0.53
        kio = kpt_iou(
            torch.from_numpy(gk), torch.from_numpy(pk),
            torch.from_numpy(gt_area), OKS_SIGMA,
        ).numpy()
        tp_p = match_predictions(pcls, gcls, kio)

        tp_l.append(tp); tp_p_l.append(tp_p)
        conf_l.append(pscores); pcls_l.append(pcls)

    def run(prefix, tp_list):
        tp = np.concatenate(tp_list)
        conf = np.concatenate(conf_l)
        pcls = np.concatenate(pcls_l)
        tcls = np.concatenate(tcls_l)
        res = ap_per_class(tp, conf, pcls, tcls, prefix=prefix)
        ap = res[5]
        m50 = np.nanmean(ap[:, 0])
        m5095 = np.nanmean(ap)
        print(f"[{tag}] {prefix} mAP50={m50:.4f} mAP50-95={m5095:.4f}")
        return m5095

    run("Box", tp_l)
    run("Pose", tp_p_l)


for t in ["fp16", "int8"]:
    evaluate(t)
print("DONE")
