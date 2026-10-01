# -*- coding: utf-8 -*-
# round5-3 INT8 AP 评测：ncnn 模型在 pose-round3 val 全集跑 decode+NMS+pycocotools keypoints AP
# 用法: python eval_ncnn_ap.py [fp16|int8|all]
import json, os, sys, time
import numpy as np
import cv2
import ncnn
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

ROOT = "/home/ubuntu/isef"
VAL_IMG = f"{ROOT}/datasets/pose-round3/images/val"
VAL_LBL = f"{ROOT}/datasets/pose-round3/labels/val"
OUT = f"{ROOT}/deploy/ncnn/pose-round5-3/ap-eval"
NAMES = ["person", "sit down", "sit up", "push down", "push up", "weights down", "weights up"]
KPT = 17
IMGSZ = 640
CONF = 0.001
IOU = 0.7
MAXDET = 300

MODELS = {
    "fp16": f"{ROOT}/runs/pose-round5-3/weights/pose-s-raw-fp16",
    "int8": f"{ROOT}/runs/pose-round5-3/weights/pose-s-raw-int8",
}

os.makedirs(OUT, exist_ok=True)


def make_anchors(hw=[(80, 80), (40, 40), (20, 20)], strides=[8, 16, 32], offset=0.5):
    anchors, strd = [], []
    for (h, w), s in zip(hw, strides):
        sx = np.arange(w, dtype=np.float32) + offset
        sy = np.arange(h, dtype=np.float32) + offset
        syy, sxx = np.meshgrid(sy, sx, indexing="ij")
        anchors.append(np.stack([sxx.ravel(), syy.ravel()], 1))
        strd.append(np.full((h * w, 1), s, dtype=np.float32))
    return np.concatenate(anchors), np.concatenate(strd)


anchors, strides = make_anchors()
anchors_T = anchors.T[None]          # [1,2,8400]
strides_T = strides.T[None]          # [1,1,8400]
ARANGE16 = np.arange(16, dtype=np.float32).reshape(1, 1, 16, 1)


def letterbox(im):
    h, w = im.shape[:2]
    r = min(IMGSZ / h, IMGSZ / w)
    uw, uh = int(round(w * r)), int(round(h * r))
    canvas = np.full((IMGSZ, IMGSZ, 3), 114, dtype=np.uint8)
    canvas[:uh, :uw] = cv2.resize(im, (uw, uh), interpolation=cv2.INTER_LINEAR)
    return canvas, uw / w


def iou(a, B):
    x1 = np.maximum(a[0], B[:, 0]); y1 = np.maximum(a[1], B[:, 1])
    x2 = np.minimum(a[2], B[:, 2]); y2 = np.minimum(a[3], B[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[2] - a[0]) * (a[3] - a[1])
    ab = (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1])
    return inter / (aa + ab - inter + 1e-16)


def build_gt(files):
    images, anns = [], []
    for i, (name, w, h) in enumerate(files):
        images.append({"id": i + 1, "file_name": name, "width": int(w), "height": int(h)})
        lp = os.path.join(VAL_LBL, os.path.splitext(name)[0] + ".txt")
        if not os.path.exists(lp):
            continue
        for line in open(lp):
            t = line.split()
            if len(t) < 5:
                continue
            t = t[:57] + ["0"] * (57 - len(t))
            t = [float(x) for x in t]
            cx, cy, bw, bh = t[1] * w, t[2] * h, t[3] * w, t[4] * h
            k = np.array(t[5:56]).reshape(KPT, 3)
            k[:, 0] *= w; k[:, 1] *= h
            anns.append({
                "id": len(anns) + 1, "image_id": i + 1,
                "category_id": int(t[0]) + 1,
                "bbox": [float(cx - bw / 2), float(cy - bh / 2), float(bw), float(bh)],
                "area": float(bw * bh), "iscrowd": 0,
                "num_keypoints": int((k[:, 2] > 0).sum()),
                "keypoints": k.flatten().tolist(),
            })
    gt = {"images": images, "annotations": anns,
          "categories": [{"id": i + 1, "name": n} for i, n in enumerate(NAMES)]}
    with open(f"{OUT}/gt.json", "w") as f:
        json.dump(gt, f)
    return gt


def decode(box, cls, kpt, ratio):
    d = box.reshape(1, 4, 16, 8400)
    e = np.exp(d - d.max(axis=2, keepdims=True))
    dist = (e / e.sum(axis=2, keepdims=True) * ARANGE16).sum(2)   # [1,4,8400]
    lt, rb = dist[:, 0:2], dist[:, 2:4]
    xy = (anchors_T - lt + anchors_T + rb) / 2.0 * strides_T       # [1,2,8400]
    wh = (rb + lt) * strides_T
    xywh = np.concatenate([xy, wh], 1)[0].T / ratio                 # [8400,4]
    sc = (1.0 / (1.0 + np.exp(-cls)))[0].T                          # [8400,7]
    k = kpt.reshape(1, KPT, 3, 8400)
    kxy = (k[:, :, :2] * 2.0 + (anchors_T.reshape(1, 1, 2, 8400) - 0.5)) * strides_T.reshape(1, 1, 1, 8400)
    kv = 1.0 / (1.0 + np.exp(-k[:, :, 2:3]))
    kp = np.concatenate([kxy, kv], 2)[0]                            # [17,3,8400]
    kp = np.transpose(kp, (2, 0, 1))                                # [8400,17,3]
    kp[:, :, :2] /= ratio
    return xywh, sc, kp


def nms(xywh, sc, kp, w, h):
    x1 = xywh[:, 0] - xywh[:, 2] / 2; y1 = xywh[:, 1] - xywh[:, 3] / 2
    x2 = xywh[:, 0] + xywh[:, 2] / 2; y2 = xywh[:, 1] + xywh[:, 3] / 2
    xyxy = np.stack([x1, y1, x2, y2], 1)
    keep = []
    for c in range(len(NAMES)):
        m = sc[:, c] > CONF
        order = np.where(m)[0][np.argsort(-sc[m, c])]
        while order.size and len(keep) < MAXDET:
            i = order[0]
            keep.append((i, c))
            rest = order[1:]
            if rest.size == 0:
                break
            order = rest[iou(xyxy[i], xyxy[rest]) < IOU]
    anns = []
    for i, c in keep:
        bw, bh = xywh[i, 2], xywh[i, 3]
        if bw <= 0 or bh <= 0:
            continue
        bx = max(0.0, min(float(xywh[i, 0] - bw / 2), w))
        by = max(0.0, min(float(xywh[i, 1] - bh / 2), h))
        bw = min(float(bw), w - bx); bh = min(float(bh), h - by)
        anns.append({
            "image_id": None, "category_id": c + 1,
            "bbox": [bx, by, bw, bh], "area": bw * bh,
            "score": float(sc[i, c]),
            "keypoints": kp[i].flatten().tolist(),
        })
    return anns


def run_model(tag, files, gt):
    net = ncnn.Net()
    net.opt.num_threads = 48
    net.load_param(MODELS[tag] + ".param")
    net.load_model(MODELS[tag] + ".bin")
    preds = []
    t0 = time.time()
    for i, (name, w, h) in enumerate(files):
        im = cv2.imread(os.path.join(VAL_IMG, name))
        canvas, ratio = letterbox(im)
        x = np.ascontiguousarray(canvas[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0)
        mat_in = ncnn.Mat(x)
        ex = net.create_extractor()
        ex.input("in0", mat_in)
        _, out0 = ex.extract("out0")
        _, out1 = ex.extract("out1")
        _, out2 = ex.extract("out2")
        box = np.asarray(out0, np.float32).reshape(1, 64, 8400)
        cls = np.asarray(out1, np.float32).reshape(1, 7, 8400)
        kpt = np.asarray(out2, np.float32).reshape(1, 51, 8400)
        xywh, sc, kp = decode(box, cls, kpt, ratio)
        for a in nms(xywh, sc, kp, w, h):
            a["image_id"] = i + 1
            a["id"] = len(preds) + 1
            preds.append(a)
        if (i + 1) % 200 == 0:
            el = time.time() - t0
            print(f"[{tag}] {i+1}/{len(files)}  {el:.0f}s  avg {el/(i+1)*1000:.0f}ms/img", flush=True)
    el = time.time() - t0
    print(f"[{tag}] DONE {len(files)} imgs in {el:.0f}s ({el/len(files)*1000:.0f}ms/img)", flush=True)
    with open(f"{OUT}/pred_{tag}.json", "w") as f:
        json.dump(preds, f)
    return el


def evaluate(tag):
    gt = COCO(f"{OUT}/gt.json")
    dt = gt.loadRes(f"{OUT}/pred_{tag}.json")
    E = COCOeval(gt, dt, "keypoints")
    E.params.imgIds = sorted(gt.getImgIds())
    E.evaluate(); E.accumulate(); E.summarize()
    print(f"[{tag}] AP= {E.stats[0]:.4f}  AP50={E.stats[1]:.4f}  AP75={E.stats[2]:.4f}", flush=True)
    return E.stats[0]


def main():
    names = sorted(os.listdir(VAL_IMG))
    files = []
    for n in names:
        im = cv2.imread(os.path.join(VAL_IMG, n))
        files.append((n, im.shape[1], im.shape[0]))
    print(f"val images: {len(files)}", flush=True)
    gt = build_gt(files)
    print(f"gt: {len(gt['images'])} imgs, {len(gt['annotations'])} anns", flush=True)
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    tags = list(MODELS) if which == "all" else [which]
    for tag in tags:
        run_model(tag, files, gt)
        evaluate(tag)


if __name__ == "__main__":
    main()
