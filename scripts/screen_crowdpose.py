#!/usr/bin/env python3
"""CrowdPose 筛选 + 转 YOLO-pose 格式（质量>数量）

筛选规则：
1. 实例 bbox 高 < 96px 剔除（远小人对居家场景是噪声）
2. 实例可见关键点(v>0) < 8/14 剔除
3. v>0 但坐标越界的关键点 → 置 v=0（脏标注降级，不扔整实例）
4. 图像解码失败剔除
关键点映射：CrowdPose 14 点 → COCO 17 点，肩肘腕髋膝踝 12 点直通，
head/neck 与 nose/eyes/ears 均置 v=0（由 COCO/MPII 数据补全这些点）
"""
import json
import os
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool

import numpy as np

SRC_JSON = "/home/ubuntu/isef/datasets/crowdpose/crowdpose_annotations/mmpose_crowdpose_trainval.json"
IMG_DIR = "/home/ubuntu/isef/datasets/crowdpose/images"
OUT_ROOT = "/home/ubuntu/isef/datasets/crowdpose/yolo"

MIN_BBOX_H = 96
MIN_VIS_KPT = 8

# CrowdPose(14) 索引 → COCO(17) 索引
MAP = {0: 5, 1: 6, 2: 7, 3: 8, 4: 9, 5: 10, 6: 11, 7: 12, 8: 13, 9: 14, 10: 15, 11: 16}

stats = Counter()


def to_yolo(ann, w, h):
    x, y, bw, bh = ann["bbox"]
    kp = ann["keypoints"]
    out = [0, (x + bw / 2) / w, (y + bh / 2) / h, bw / w, bh / h]
    for i in range(17):
        out.extend([0.0, 0.0, 0.0])
    for src, dst in MAP.items():
        px, py, v = kp[src * 3], kp[src * 3 + 1], kp[src * 3 + 2]
        if v > 0 and (px < 0 or px > w or py < 0 or py > h):
            v = 0
        if v > 0:
            out[5 + dst * 3] = px / w
            out[5 + dst * 3 + 1] = py / h
            out[5 + dst * 3 + 2] = v
    return " ".join(f"{t:.6f}" for t in out)


def process_image(item):
    img_id, anns, dims = item
    w, h = dims
    try:
        from PIL import Image
        with Image.open(os.path.join(IMG_DIR, f"{img_id}.jpg")) as im:
            im.verify()
    except Exception:
        return img_id, "img_bad", None
    lines = [to_yolo(a, w, h) for a in anns]
    return img_id, "ok", lines


def main():
    d = json.load(open(SRC_JSON))
    imgs = {im["id"]: im for im in d["images"] if im["width"] > 0 and im["height"] > 0}
    anns_by_img = defaultdict(list)
    for a in d["annotations"]:
        if a["image_id"] not in imgs:
            stats["ann_no_img"] += 1
            continue
        if a.get("iscrowd", 0):
            stats["ann_crowd"] += 1
            continue
        x, y, bw, bh = a["bbox"]
        if bw <= 0 or bh <= 0:
            stats["ann_bad_bbox"] += 1
            continue
        if bh < MIN_BBOX_H:
            stats["ann_small"] += 1
            continue
        kp = a["keypoints"]
        if sum(1 for v in kp[2::3] if v > 0) < MIN_VIS_KPT:
            stats["ann_few_vis"] += 1
            continue
        stats["ann_kept"] += 1
        anns_by_img[a["image_id"]].append(a)

    items = [(iid, anns_by_img[iid], (imgs[iid]["width"], imgs[iid]["height"]))
             for iid in imgs if anns_by_img.get(iid)]

    os.makedirs(os.path.join(OUT_ROOT, "images", "train"), exist_ok=True)
    os.makedirs(os.path.join(OUT_ROOT, "labels", "train"), exist_ok=True)

    with Pool(32) as pool:
        results = pool.map(process_image, items)

    kept = 0
    for img_id, status, lines in results:
        if status == "img_bad":
            stats["img_bad"] += 1
            continue
        if not lines:
            stats["img_no_ann"] += 1
            continue
        label = os.path.join(OUT_ROOT, "labels", "train", f"{img_id}.txt")
        with open(label, "w") as f:
            f.write("\n".join(lines) + "\n")
        src = os.path.join(IMG_DIR, f"{img_id}.jpg")
        dst = os.path.join(OUT_ROOT, "images", "train", f"{img_id}.jpg")
        if not os.path.exists(dst):
            os.symlink(src, dst)
        kept += 1

    print("=== 筛选报告 ===")
    print("源图像:", len(imgs), "源标注:", len(d["annotations"]))
    for k in ["ann_no_img", "ann_crowd", "ann_bad_bbox", "ann_small", "ann_few_vis", "ann_kept"]:
        print(f"  {k}: {stats[k]}")
    print("图像级: 解码失败", stats["img_bad"], "无有效实例", stats["img_no_img"] if "img_no_img" in stats else stats["img_no_ann"])
    print("保留图像:", kept, " 保留标注:", stats["ann_kept"])


if __name__ == "__main__":
    main()
