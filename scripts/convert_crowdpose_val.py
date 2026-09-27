#!/usr/bin/env python3
"""convert_crowdpose_val.py — CrowdPose val（2k 图）转 YOLO-pose 格式，评测用不做筛选"""
import json
import os

SRC_JSON = "/home/ubuntu/isef/datasets/crowdpose/crowdpose_annotations/mmpose_crowdpose_val.json"
IMG_DIR = "/home/ubuntu/isef/datasets/crowdpose/images"
OUT_ROOT = "/home/ubuntu/isef/datasets/crowdpose/yolo-val"

MAP = {0: 5, 1: 6, 2: 7, 3: 8, 4: 9, 5: 10, 6: 11, 7: 12, 8: 13, 9: 14, 10: 15, 11: 16}


def to_yolo(ann, w, h):
    x, y, bw, bh = ann["bbox"]
    kp = ann["keypoints"]
    out = [0, (x + bw / 2) / w, (y + bh / 2) / h, bw / w, bh / h]
    out.extend([0.0, 0.0, 0.0] * 17)
    for src, dst in MAP.items():
        px, py, v = kp[src * 3], kp[src * 3 + 1], kp[src * 3 + 2]
        if v > 0 and (px < 0 or px > w or py < 0 or py > h):
            v = 0
        if v > 0:
            out[5 + dst * 3] = px / w
            out[5 + dst * 3 + 1] = py / h
            out[5 + dst * 3 + 2] = v
    return " ".join(f"{t:.6f}" for t in out)


def main():
    d = json.load(open(SRC_JSON))
    imgs = {im["id"]: im for im in d["images"]}
    os.makedirs(os.path.join(OUT_ROOT, "images", "val"), exist_ok=True)
    os.makedirs(os.path.join(OUT_ROOT, "labels", "val"), exist_ok=True)
    n = 0
    for a in d["annotations"]:
        im = imgs[a["image_id"]]
        path = os.path.join(OUT_ROOT, "labels", "val", f"{a['image_id']}.txt")
        with open(path, "a") as f:
            f.write(to_yolo(a, im["width"], im["height"]) + "\n")
    for iid, im in imgs.items():
        src = os.path.join(IMG_DIR, f"{iid}.jpg")
        dst = os.path.join(OUT_ROOT, "images", "val", f"{iid}.jpg")
        if os.path.exists(src) and not os.path.exists(dst):
            os.symlink(src, dst)
            n += 1
    print("val 图像软链:", n, "标注实例:", len(d["annotations"]))


if __name__ == "__main__":
    main()
